# PyIB 0.1.0rc2

PyIB is a Python immersed-boundary solver with CPU and CUDA backends.
Repository: <https://github.com/XuZhaoyue1995/PyIB>.
Sole software author: **Zhaoyue Xu**.

**Release scope: the CPU/CUDA solver, pressure-field reconstruction and
surface-pressure correction.** Solver and pressure numerical validation
has been completed on the D32/D40 oscillating-sphere benchmark (Re₀ = 0.2).
The release includes the measured errors and their supporting evidence.

## Current postprocessing — 2026-09-25

Pressure reconstruction and surface-pressure correction retain their
validated numerical implementation. Surface integration reports pressure
force only. Wall-shear recovery and surface-total-force output are excluded
from this release. See [postprocessing](docs/POSTPROCESSING.md).

D32/D40 simulations completed six periods using CUDA and float64. Runtime
force harmonic-vector errors are 5.2690% / 4.1190%; corrected surface-pressure
relative area/time L2 errors are 2.5300% / 2.2224%. Periodic convergence,
joint grid/time refinement and source identities are documented in
[VALIDATION.md](docs/VALIDATION.md). The
[validation summary](checks/pressure-validation-20260925.json) and
[evidence archive](checks/pressure-validation-20260925.zip) accompany the release.

## Distribution and attribution

`SOURCE_MANIFEST.json` verifies the current numerical files. Thirteen files
retain their captured source bytes; the pressure-only postprocessor has a
new hash and an explicit record of its original source hash. The CPU/CUDA
numerical time integrator is unchanged. Installation and commands are described in
[README.md](README.md).

The original manuscript in `paper/IB_GPU_Solver_Paper_CN_JCP.pdf` remains
unchanged, including its original author information. Software attribution is
recorded in `AUTHORS.md` and `CITATION.cff`.

The software license remains pending the author's selection. No open-source
license has been granted by this distribution.
