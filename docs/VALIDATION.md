# Solver and pressure validation: D32/D40 oscillating sphere

Author: **Zhaoyue Xu**. Results retrieved and independently reviewed on
2026-09-25. The D32 and D40 simulations completed by 2026-09-23.

**The solver and pressure modules have completed documented numerical
validation for this benchmark.** The checks combine analytical comparison,
periodic convergence, joint grid/time refinement and source/data provenance.
The release ships the solver, pressure-field reconstruction and
surface-pressure correction. The pressure accuracy figures below measure
the reconstructed and corrected surface pressure.

## Problem and source identity

A sphere of radius 0.25 oscillates along the x axis in initially stationary
fluid. Density is 1, peak speed is 0.02, angular frequency is 3.6, kinematic
viscosity is 0.05, Re₀ = U₀(2R)/ν = 0.2, R/δ = 1.5 and KC = 0.06981317.
Each run contains six periods. The reference is the linear, unbounded
unsteady-Stokes solution; the numerical case uses finite-Re Navier–Stokes
in a finite domain.

| Run | Cells | Steps per period | Total steps | Scheduler result |
|---|---:|---:|---:|---|
| D32, job 251991 | 1,179,648 | 560 | 3,360 | COMPLETED, exit 0 |
| D40, job 252004 | 2,304,000 | 700 | 4,200 | COMPLETED, exit 0 |

Both runs used CUDA, float64 and four inner iterations. The simulation source
files remain identical to the frozen campaign. The pressure-only offline
postprocessor retains the same pressure reconstruction and surface-pressure
correction calculations. Its current and original hashes are distinguished
in `SOURCE_MANIFEST.json`. The measured D32/D40 results are CUDA results;
the package also provides the NumPy/SciPy CPU backend.

## Measured pressure and runtime-force results

| Quantity | D32 | D40 |
|---|---:|---:|
| Runtime body-force harmonic-vector error / analytic force amplitude | 5.2690% | 4.1190% |
| Runtime body-force amplitude error | +5.0513% | +3.9570% |
| Phase error, atan2(sine, cosine) convention | −0.8380° | −0.6428° |
| Cycle 5→6 waveform RMS difference / analytic force amplitude | 0.034108% | 0.033814% |
| Corrected surface-pressure relative area/time L2 error | 2.5300% | 2.2224% |

The force fits use the final two cycles. The force harmonic-vector error is
the Euclidean difference between fitted numerical and analytical cosine/sine
coefficients, divided by the analytical force amplitude. The runtime body
force includes the internal-fluid momentum-rate correction. Its harmonic
coefficient vector changes by 1.1508% of the analytic force amplitude between
D32 and D40. This force is read from the solver history; it is not a
surface-pressure-only force or a reconstructed total surface force.

Surface pressure was evaluated at 5T, 5.25T, 5.5T, 5.75T and 6T. The four
distinct phases supply the area/time norm, using marker-area weights and the
analytical squared-field integral as denominator. For D40, five-phase
pressure RMS divided by the phase-specific analytic spatial maximum falls
from 22.23–48.98% raw to 1.21–1.36% corrected. The same saved far-shell
reference shift is used for raw and corrected surface pressure.

Grid spacing and time step both decrease by a factor 0.8; this dataset
therefore measures joint grid/time refinement. The reported error levels
apply to this Re₀ = 0.2 oscillating-sphere setup and the stated resolutions;
they are not an accuracy estimate for a stationary sphere at Re = 200.

## Checks and examples

```sh
python tools/verify_source_manifest.py
python tools/run_tests.py
python tools/check_pressure_validation.py
```

The [machine-readable validation summary](../checks/pressure-validation-20260925.json)
and [evidence archive](../checks/pressure-validation-20260925.zip) accompany
the release. Source hashes link the retained solver and pressure methods to
the evaluated code. The bundled small configurations in `examples/`
exercise installation and case execution; they serve a different purpose
from the physical benchmark above.

For the distinction between unsteady-Stokes and finite-Re oscillating-sphere
references, see Mei (1994), [Flow due to an oscillating sphere and an expression
for unsteady drag on the sphere at finite Reynolds number](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/flow-due-to-an-oscillating-sphere-and-an-expression-for-unsteady-drag-on-the-sphere-at-finite-reynolds-number/13386D6EC497ADA02ADDDD5A750B2609).
