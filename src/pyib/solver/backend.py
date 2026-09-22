"""Array backend: NumPy (CPU, default) or CuPy (CUDA, set IB_GPU=1).

CPU and GPU share numerical source through ``xp``; this does not imply bitwise
equality or validation on every device. Mesh generation stays on the host;
IBCore transfers numerical arrays to the selected backend. Select the device
and precision before importing solver modules, preferably via cfdagent_ib.py.
"""
import os

GPU = os.environ.get("IB_GPU", "0") == "1"
# FP32 halves floating-array storage (not index storage). Whole-solver speed and
# accuracy depend on hardware, conditioning, grid and timestep; validate them
# per case. Float64 is the default for scientific comparisons. CG tolerances
# are adjusted in the driver for float32, so trajectories need not be identical.
FP32 = os.environ.get("IB_FP32", "0") == "1"

if GPU:                                   # single CUDA device
    import cupy as xp
    import cupyx.scipy.sparse as xsp
    import cupyx.scipy.sparse.linalg as xspla

    def asnumpy(a):
        return xp.asnumpy(a)
else:                                     # default CPU path
    import numpy as xp
    import scipy.sparse as xsp
    import scipy.sparse.linalg as xspla

    def asnumpy(a):
        import numpy as _np
        return _np.asarray(a)


real = xp.float32 if FP32 else xp.float64   # working float dtype


def asarray(a):
    return xp.asarray(a)


def asreal(a):
    """To the active backend, casting FLOAT arrays to the working dtype (real);
    integer/bool index arrays pass through unchanged."""
    a = xp.asarray(a)
    return a.astype(real, copy=False) if a.dtype.kind == "f" else a
