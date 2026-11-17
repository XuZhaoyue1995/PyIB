# Optional CPU CSR acceleration

PyIB defaults to SciPy sparse multiplication. A Linux OpenMP extension is an
explicit option for CPU float64 runs. It assigns independent CSR rows to
threads while retaining each row's stored accumulation order. The matrix,
right-hand side, CG algorithm, tolerances, time step, IB coupling, and pressure
methods are unchanged. Matrix shape and geometry are not hardcoded. Native
int32 and int64 CSR indices are supported without copying the full matrix.

## `threads` backend: no compiler needed (added 2026-11-17)

```sh
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/threads --cpu-sparse-backend threads --cpu-sparse-threads 8
```

JSON: `"cpu_sparse_backend": "threads"`, `"cpu_sparse_threads": 8`.  The rows of
each CSR matrix are cut into blocks of equal non-zero count and SciPy's own CSR
kernel is applied to the blocks on a thread pool (the kernel releases the
interpreter lock).  Every row keeps its stored accumulation order, so the
product equals the SciPy product bit for bit; this is checked once when the
operator is built.  It works on any platform and with either precision, and with
both forms of the operator (`solver.operator`).  The kernel is called through
SciPy's private `_sparsetools` module; if a future SciPy changes it, the check at
construction fails with a clear error instead of giving wrong numbers.

Measured on the D16 oscillating-sphere case on an i9-9900K (shared machine,
single runs, first 3 steps): factored operator 125.2 s with SciPy, 100.5 s with
8 threads; assembled operator 238.4 s with 8 threads against 450.0 s for the
as-found code.  The sparse products are bound by memory bandwidth, so the gain
saturates at a few threads.  Details in [PERFORMANCE.md](PERFORMANCE.md).

As with `openmp`, the backend and thread count are part of the recorded
execution identity: a checkpoint written with one setting is not resumed with
another.

The rest of this document describes the `openmp` extension of 2026-10-05.  For
the 2026-11-17 changes it was rebuilt once on WSL Ubuntu 22.04 (GCC 11.4, Python
3.10, NumPy 2.1.2, SciPy 1.14.1) from a scratch copy of the source tree: all 71
unit tests passed with `PYIB_REQUIRE_OPENMP_TESTS=1` (none skipped), and on
`cpu_fixed_sphere_re100_d8_gate` with the factored operator both `openmp` and
`threads` with 4 threads gave `history.csv` and final fields bit-identical to
SciPy.  The compute-node regression of 2026-10-05 described below was not
repeated.  (`python setup.py build_ext --inplace` builds the extension but fails
to copy it into the `src` layout; `pip install .` is the supported route.)

## Build and select

```sh
PYIB_BUILD_OPENMP=1 python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/parallel --cpu-sparse-backend openmp --cpu-sparse-threads 4
```

The equivalent JSON solver options are `"cpu_sparse_backend": "openmp"` and
`"cpu_sparse_threads": 4`. Omission selects `scipy` and 1. OpenMP requires a
Linux GCC-compatible compiler and runtime; ordinary installation stays pure
Python. The package does not bind to a particular cluster, CPU numbering or
partition. Users choose the worker count within their allocation. Speedup
depends on matrix size, memory bandwidth and other work in each step.

An explicit OpenMP request fails clearly when unavailable, on CPU float32,
on CUDA, or with the matrix-free path. It never triggers compilation, downloads
or a silent fallback. CUDA continues through its existing CuPy path.

## Reproducibility and restart

Result schema 6 and checkpoint schema 4 record `sparse_execution` in metadata
and `sparse_execution_json` in NPZ artifacts. This includes the selected backend,
thread count, observed OpenMP team, relevant compiled implementation hashes and
build information. Each application checks the actual team and buffer bounds.

Restart and validation require the same recorded source and execution identity.
Changing threads, implementation or binary requires a separate, explicitly
verified migration; editing a checkpoint's identity is not a supported migration.
Older schema 5/3 results remain usable with their original PyIB release. This
revision does not rewrite them. The artifact validator checks identities without
building a mesh or running a sparse multiplication.

## Verification scope

Before this integration, an isolated D24 experiment on paraCPU measured a
251.504 s SciPy step versus a 113.494 s four-thread prototype step (2.216 times
faster). All saved arrays and four inner CG iteration counts matched exactly.
That was one step using an external prototype, not a speed guarantee for this
packaged implementation or all geometries and computers.

The maintained implementation passed its compute-node regression on 2026-10-05
(Slurm job 12259735, Linux/GCC 12.2, Python 3.10, NumPy 2.2.6, SciPy 1.15.3):

- All 23 compiled unit tests passed, including pressure, CSR safety, both index
  widths, 1/2/4 threads, default selection and restart-identity rejection.
- Fixed and oscillating spheres each ran twelve steps with the original SciPy,
  updated SciPy, OpenMP1 and OpenMP4 implementations. All ten native output
  validators passed.
- All ten comparisons passed with zero reported numerical differences:
  656 stored-member and 400 history-column comparisons. Every one of the 1,026
  recorded bitwise checks was identical; the remaining 30 integer checks were
  exact. Timing and differing run/source identities are explicitly excluded.
- For each geometry, six steps plus native restart to twelve matched the
  uninterrupted OpenMP4 run. That direct comparison excludes only the run UUID.

Observed elapsed times for a complete twelve-step process, including setup
and output, were:

| Short test | Updated SciPy | OpenMP, 4 threads |
|---|---:|---:|
| Fixed sphere | 5.982 s | 2.822 s |
| Oscillating sphere | 6.186 s | 2.922 s |

These are single ordered trials. The checks establish implementation and restart
consistency for the tested CPU environment; the short coarse examples do not
measure physical accuracy or general speedup. GPU execution and clean wheel
installation were not part of this source-tree regression.

The [test record](../checks/cpu-csr-regression-20261005.json) includes the exact
configurations, source/binary identities and unit output. The
[complete comparisons](../checks/cpu-csr-comparison-20261005.json) retain all
per-member differences. The historical D32/D40 physical validation is unchanged.

For developers, `python tools/run_tests.py` exercises the available backends.
Set `PYIB_REQUIRE_OPENMP_TESTS=1` after building to require the compiled tests
instead of allowing the optional extension tests to skip.
