"""Optional CPU CSR backends and durable execution identity.

SciPy is the default. An explicit OpenMP request never compiles, installs or
falls back. The extension owns no solver state and changes only row scheduling.

``threads`` needs no compiled extension: the rows are cut into blocks of equal
non-zero count and SciPy's own CSR kernel is applied to the blocks on a thread
pool (the kernel releases the interpreter lock).  Every row keeps its stored
accumulation order, so the result equals the SciPy product bit for bit.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import importlib

import numpy as np


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extension():
    name = (__package__ + "." if __package__ else "") + "_csr_openmp"
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "CPU OpenMP was explicitly requested but its optional extension is unavailable. "
            "Build/install PyIB on Linux with PYIB_BUILD_OPENMP=1 and a GCC-compatible "
            "OpenMP compiler, or explicitly select cpu_sparse_backend='scipy'. "
            "No runtime build or fallback was attempted."
        ) from exc


def validate_options(backend, threads, *, device="cpu", precision="float64", sparse=True):
    if not isinstance(backend, str) or backend not in {"scipy", "threads", "openmp"}:
        raise ValueError("solver.cpu_sparse_backend must be 'scipy', 'threads' or 'openmp'")
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 2147483647:
        raise ValueError("solver.cpu_sparse_threads must be a positive int32 integer")
    if backend == "scipy" and threads != 1:
        raise ValueError("cpu_sparse_threads needs cpu_sparse_backend 'threads' or 'openmp'; use 1 with the SciPy default")
    if backend == "threads" and (device != "cpu" or not sparse):
        raise ValueError("threaded CSR requires runtime.device='cpu' and solver.sparse=true")
    if backend == "threads" and threads > 256:
        raise ValueError("solver.cpu_sparse_threads must be <= 256 for the threads backend")
    if backend == "openmp" and (device != "cpu" or precision != "float64" or not sparse):
        raise ValueError("OpenMP CSR requires runtime.device='cpu', precision='float64' and solver.sparse=true")


def execution_identity(config, *, check_team=False):
    """Identify the selected implementation without building a mesh/operator.

Paths are resolved to hash their actual files, but are not part of restart
identity: relocation of identical binaries is permitted. Only directly used
CSR implementation and OpenMP binaries are included, not every system library.
"""
    solver, runtime = config["solver"], config["runtime"]
    backend, threads = solver["cpu_sparse_backend"], solver["cpu_sparse_threads"]
    validate_options(backend, threads, device=runtime["device"], precision=runtime["precision"], sparse=solver["sparse"])
    actual = "matrix_free" if not solver["sparse"] else ("cupy" if runtime["device"] == "cuda" else backend)
    result = {"schema": 1, "backend": actual,
              "threads": int(threads) if actual in {"openmp", "threads"} else 1,
              "actual_threads": None, "implementation_sha256": {}, "build": {}}
    if actual in {"scipy", "threads"}:
        import scipy
        from scipy.sparse import _sparsetools, _compressed
        result["implementation_sha256"] = {"scipy_extension": _sha(_sparsetools.__file__),
                                            "scipy_dispatcher": _sha(_compressed.__file__)}
        result["build"] = {"numpy": np.__version__, "scipy": scipy.__version__}
        if actual == "threads":
            result["actual_threads"] = int(threads)
    elif actual == "openmp":
        module = _extension()
        info = dict(module.build_info())
        library = Path(info.pop("runtime_path")).resolve(strict=True)
        if info.get("abi") != 1:
            raise RuntimeError("Unsupported PyIB OpenMP extension ABI")
        result["implementation_sha256"] = {"extension": _sha(module.__file__), "openmp_runtime": _sha(library)}
        result["build"] = info
        # Runtime construction observes a real team before writing metadata.
        # Artifact validation compares identity only and does not run a kernel.
        actual = int(threads)
        if check_team:
            pointers = np.zeros(1, dtype=np.int32)
            indices = np.empty(0, dtype=np.int32)
            empty = np.empty(0, dtype=np.float64)
            actual = module.apply(0, 0, pointers, indices, empty, empty, empty, int(threads))
        result["actual_threads"] = actual
    return result


class ThreadedCSR:
    """SciPy CSR product on row blocks, one block per thread.

The driver owns the matrix; this object keeps views of its buffers, a thread
pool and per-block row pointers.  ``matrix @ vector`` equals SciPy bit for bit.
"""
    def __init__(self, matrix, threads):
        from scipy.sparse import isspmatrix_csr, csr_array, _sparsetools
        validate_options("threads", threads)
        if not (isspmatrix_csr(matrix) or isinstance(matrix, csr_array)):
            raise ValueError("threaded CSR requires an existing CSR matrix/array")
        if matrix.dtype not in (np.dtype("float64"), np.dtype("float32")) or not matrix.dtype.isnative:
            raise ValueError("threaded CSR requires native float32/float64 coefficients")
        self._matrix = matrix
        self._kernel = _sparsetools.csr_matvec
        self.shape, self.dtype = matrix.shape, matrix.dtype
        self.nnz = int(matrix.nnz)
        rows = matrix.shape[0]
        self.threads = int(max(1, min(int(threads), rows)))
        cuts = np.searchsorted(matrix.indptr, np.linspace(0, matrix.nnz, self.threads + 1)).astype(np.int64)
        cuts[0], cuts[-1] = 0, rows
        cuts = np.maximum.accumulate(cuts)
        self._blocks = []
        for first, last in zip(cuts[:-1], cuts[1:]):
            if last == first:
                continue
            lo, hi = int(matrix.indptr[first]), int(matrix.indptr[last])
            pointers = np.ascontiguousarray(matrix.indptr[first:last + 1] - matrix.indptr[first])
            self._blocks.append((int(first), int(last), pointers, matrix.indices[lo:hi], matrix.data[lo:hi]))
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self._blocks)),
                                        thread_name_prefix="pyib-csr")
        # One-time self-check against SciPy: the private kernel signature is an
        # assumption about the installed SciPy, not a documented interface.
        probe = np.cos(np.arange(self.shape[1], dtype=np.float64)).astype(self.dtype)
        if not np.array_equal(self @ probe, matrix @ probe):
            raise RuntimeError("threaded CSR self-check failed: result differs from the SciPy product")

    def __matmul__(self, vector):
        if not isinstance(vector, np.ndarray) or vector.dtype != self.dtype or vector.ndim != 1:
            raise ValueError("threaded CSR vector must be a 1-D ndarray of the matrix dtype")
        if vector.shape != (self.shape[1],) or not vector.flags.c_contiguous:
            raise ValueError("threaded CSR vector must be contiguous and match the column count")
        out = np.zeros(self.shape[0], dtype=self.dtype)
        columns, kernel = self.shape[1], self._kernel

        def apply(block):
            first, last, pointers, indices, data = block
            kernel(last - first, columns, pointers, indices, data, vector, out[first:last])

        if len(self._blocks) == 1:
            apply(self._blocks[0])
        else:
            list(self._pool.map(apply, self._blocks))
        return out


class OpenMPCSR:
    """Read-only references to a native SciPy CSR, without a matrix copy.

The driver owns the CSR. External concurrent mutation is unsupported. C still
checks row pointers and every column index during every apply, so a stale
Python validation or writable alias cannot bypass buffer bounds checks.
"""
    def __init__(self, matrix, threads):
        from scipy.sparse import isspmatrix_csr, csr_array
        validate_options("openmp", threads)
        if not (isspmatrix_csr(matrix) or isinstance(matrix, csr_array)):
            raise ValueError("OpenMP requires an existing CSR matrix/array")
        if matrix.dtype != np.dtype("float64") or not matrix.dtype.isnative:
            raise ValueError("OpenMP requires native FP64 CSR coefficients")
        if matrix.indices.dtype not in (np.dtype("int32"), np.dtype("int64")) or matrix.indptr.dtype != matrix.indices.dtype:
            raise ValueError("OpenMP requires matching native int32/int64 CSR indices")
        for value in (matrix.data, matrix.indices, matrix.indptr):
            if value.ndim != 1 or not value.flags.c_contiguous or not value.flags.aligned:
                raise ValueError("OpenMP CSR buffers must be aligned and contiguous")
        # Shape/pointers/index bounds are also validated by the extension on
        # every application, with no extra scan/copy of the full matrix.
        self._matrix = matrix
        self.shape, self.dtype = matrix.shape, matrix.dtype
        self.threads = int(threads)
        self._extension = _extension()
        self.last_actual_threads = None

    def __matmul__(self, vector):
        if not isinstance(vector, np.ndarray) or vector.dtype != np.dtype("float64") or not vector.dtype.isnative:
            raise ValueError("OpenMP CSR vector must be a native FP64 ndarray")
        if vector.ndim != 1 or vector.shape != (self.shape[1],) or not vector.flags.c_contiguous or not vector.flags.aligned:
            raise ValueError("OpenMP CSR vector must be aligned, contiguous and match the column count")
        out = np.zeros(self.shape[0], dtype=np.float64)
        self.last_actual_threads = self._extension.apply(
            *self.shape, self._matrix.indptr, self._matrix.indices,
            self._matrix.data, vector, out, self.threads,
        )
        return out
