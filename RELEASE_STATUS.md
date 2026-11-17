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

## Working tree after 0.1.0rc2 — 2026-11-17, not released

The source tree contains changes that are not part of the `0.1.0rc2` release
and have not been tagged or released: a factored Helmholtz operator
and in-place CG (faster, smaller), an optional K-matrix force correction (off by
default), a two-sided surface-pressure correction for triangulated surfaces, a
`threads` CPU backend and an optional floor of the CG stopping test.  The
version number is unchanged.  What was measured and what was not is in
[CHANGELOG.md](CHANGELOG.md), [docs/PERFORMANCE.md](docs/PERFORMANCE.md),
[docs/K_MATRIX.md](docs/K_MATRIX.md) and [docs/VALIDATION.md](docs/VALIDATION.md).
The D32/D40 results above were produced by the released code; they were not
rerun in full with the changed code.

## Distribution and attribution

`SOURCE_MANIFEST.json` verifies current numerical files and retains original
hashes for changed files. Optional CPU CSR parallel multiplication and explicit
execution identity have been added; the CG/time-stepping and pressure methods
are retained. The historical D32/D40 evidence retains its original source
identities. Installation and commands are described in
[README.md](README.md).

The original manuscript in `paper/IB_GPU_Solver_Paper_CN_JCP.pdf` remains
unchanged, including its original author information. Software attribution is
recorded in `AUTHORS.md` and `CITATION.cff`.

The source tree is licensed under the BSD 3-Clause License ([LICENSE](LICENSE))
since 2026-11-17. The `0.1.0rc2` release was published before the license was
chosen and carried none. The license covers the software only; the manuscript in
`paper/` is excluded ([note](paper/README.md)).
