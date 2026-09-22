# Dependency and portability boundary

PyIB — A Python Immersed Boundary Solver for CPUs and GPUs — exposes the
`pyib` namespace and `pyib-run`, `pyib-postprocess`, `pyib-validate` commands.
The internal historical `solver/cfdagent_ib.py` filename and log prefix remain
part of the byte-preserved numerical source snapshot; they do not set the
public package name.

The 14 preserved files are listed with SHA-256 hashes in
`SOURCE_MANIFEST.json`. The mesh generator remains one directory above the
solver modules, preserving the numerical source hashing and imports expected
by the existing runner and validator. The console wrappers insert those two
directories into the process import path before dispatching to the existing
`main` functions. Use one device and precision per process.

Standard runtime dependencies follow the source requirements exactly:
NumPy >=2,<3; SciPy >=1.13,<2; h5py >=3.11,<4; meshio >=5.3,<6.
The optional GPU dependency is cupy-cuda12x >=13,<14, requiring a compatible
CUDA 12/NVIDIA environment. The CUDA kernels in the solver are embedded Python
strings; no external CUDA source or compiled project library is needed.

No external geometry/data file is required by the four included sphere case
definitions. Dynamic sphere mesh imports (`sphere_wake_mesh` and
`sphere_quiescent_mesh`) are included. Offline postprocessing reads explicitly
provided NPZ snapshots and uses NumPy/SciPy on the CPU, including for snapshots
produced on CUDA. The validator reads a user-selected run directory and the
packaged numerical files for provenance checks.

The mesh generator also contains an optional `pymetis` import for partition
export. That historical function is not used by the single-domain case runner,
is not exposed by the console commands, and is not a required dependency here.
MPI and standalone historical demo entry points are outside this package's
supported interface.

Four preserved `__main__` diagnostic blocks contain the developer's historical
local Windows project path: `ib_sparse.py`, `ib_driver_np.py`,
`ib_immersed.py`, and `ib_core_np.py`. Those paths are not accessed by normal
imports or the packaged CLI. They remain unchanged to preserve provenance;
directly running those modules as scripts is unsupported. The mesh generator
also retains its original author attribution. A separate reviewed cleanup can
remove obsolete demo blocks before public release, with an updated manifest.

The stable interface for this candidate is the three console commands.
Importing arbitrary legacy modules into a larger Python process may collide
with generic names such as `backend`; process isolation is recommended until
the imports receive a separately tested package refactor.
