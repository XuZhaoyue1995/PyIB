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

Both runs used CUDA, float64 and four inner iterations. Their original simulation
sources and identities remain in the frozen campaign evidence. The maintained
source now also includes optional CPU CSR execution and stronger restart
identity; those changes do not retroactively alter the recorded runs. The pressure-only offline
postprocessor retains the same pressure reconstruction and surface-pressure
correction calculations. Its current and original hashes are distinguished
in `SOURCE_MANIFEST.json`. The measured D32/D40 results are CUDA results;
CPU implementation regression is documented separately in
[CPU acceleration](CPU_ACCELERATION.md).

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

## Checks of the 2026-11-17 working tree (unreleased changes)

The results above were produced by the released `0.1.0rc2` code on a cluster
GPU.  The working tree of 2026-11-17 changes how the Helmholtz operator is
applied (three factors instead of one assembled matrix, in-place CG updates) and
adds an optional K-matrix force correction and a two-sided pressure correction
([CHANGELOG.md](../CHANGELOG.md)).  The full six-period D32 and D40 runs were
**not** repeated with the changed code: on the available workstation (RTX 2080,
8 GB) D32 alone takes about 12 hours.  What was run instead, all on that
workstation in float64, single runs, is listed here; the machine-readable record
is [checks/upgrade-20261117.json](../checks/upgrade-20261117.json).

### The changed code against the archived D32 run

The D32 case of the evidence archive was started again with the changed code
(default factored operator, CUDA, CuPy 13.6; the archived run used CuPy 14.1 on
another GPU) and compared with the archived `history.csv` at the steps both
contain.

| Quantity at the common steps 1, 2, 3, 5, 10 | Largest difference / largest archived value |
|---|---:|
| Body force (x) | 3.5e-14 |
| Force on the fluid (x) | 5.2e-15 |
| Interior-fluid momentum (x) | 2.7e-15 |
| Marker slip (rms) | 5.4e-15 |
| Largest speed | 1.3e-13 |
| CG iterations of the largest inner solve | identical (2220, 2220, 2220, 2220, 1820) |

The run was stopped by hand after step 10: it shared the GPU with two other
validation runs and needed about two minutes per step.  An earlier 12-step run
of the same case alone on the GPU (468 s) gave the same agreement (1.9e-14 in
the body force).  This shows that the changed code follows the archived
trajectory through the start-up; it does **not** show the six-period error
figures of the tables above for the changed code.  Those would follow if the
agreement persisted over all 3,360 steps, which was not run.

### Same discrete case in IBFast

Sphere oscillating in fluid at rest, D = 8h, 201 markers, 560 steps (two
periods), `cg_check_interval = 1`.  IBFast runs: explicit update of 2026-09-22
(`cases/paper_osc_d8`), constrained solve of 2026-11-16
(`tests/osc_sphere_pressure.sh`); they were not rerun.

| Comparison | Largest difference of the force on the fluid over the 560 steps / largest force |
|---|---:|
| PyIB explicit update against IBFast explicit update | 2.7e-7 |
| PyIB K-matrix correction against IBFast constrained solve (`ib_update='pcg'`) | 3.4e-10 |

The force written by IBFast has 12 significant digits.  Surface pressure of the
final field (zero-velocity phase, t = 2T), RMS error over the markers divided by
the largest analytic surface pressure of that phase, with the pressure zero from
the exterior shell 4h–6h:

| | raw | corrected |
|---|---:|---:|
| PyIB explicit update, sphere-exterior correction | 25.9% | 3.29% |
| PyIB K-matrix, sphere-exterior correction | 25.8% | 3.44% |
| PyIB K-matrix, two-sided correction, outer side | 26.3% | 3.37% |
| IBFast explicit update (its changelog) | 26.6% | 3.2% |
| IBFast constrained solve (its changelog) | 26.4% | 3.3% |

### Two-sided pressure correction against the IBFast tool

`ib_pressure_surface.two_sided_surface_pressure` and IBFast
`tools/pressure_lfc.py --two-sided` (commit `8aac51b`) were given the same IBFast
dumps.  Largest difference of the marker pressures (both sides, raw and
corrected) divided by the largest corrected pressure:

| Dump | Surface | Difference |
|---|---|---:|
| Impulsively started zero-thickness disc, D/h = 16, one constraint per marker, kernel h3s | open, 263 markers | 4.0e-12 |
| Same, constraint patches | open | 5.1e-15 |
| Same, D/h = 32, constraint patches | open | 6.0e-15 |
| Oscillating sphere D = 8h, kernel h3 | closed, 201 markers | 2.7e-15 |
| Same with `far_gauge = 0.3` | closed | 1.5e-11 |

The faces cut by the surface, the band cells and the triangle orientation are
identical in every case.  The accuracy of the method on the disc against the
potential-flow solution was measured by the author with the IBFast tool and is
kept with the IBFast records; since the two implementations agree to round-off
on these inputs it was not measured again here.

### D16, six periods: explicit update and K-matrix correction against the analytic solution

The benchmark of this document at 16 cells per diameter (147,456 cells, 804
markers, 280 steps per period, 1,680 steps, four inner iterations, CUDA,
float64, factored operator, `cg_check_interval = 20`), run twice with the
changed code: with the explicit update and with the K-matrix correction (one
constraint per marker).  Same definitions as in the tables above: force fit over
the last two periods, surface pressure at 5T, 5.25T, 5.5T, 5.75T (and 6T as the
repeated phase), pressure zero from the exterior shell 4h–6h recomputed from the
recovered field.  There is no archived D16 run to compare with; the D32 and D40
figures of the released code are repeated for the trend.

| Quantity | D16 explicit | D16 K-matrix | D32 (archived) | D40 (archived) |
|---|---:|---:|---:|---:|
| Runtime body-force harmonic-vector error | 11.69% | 12.15% | 5.2690% | 4.1190% |
| Runtime body-force amplitude error | +11.14% | +11.51% | +5.0513% | +3.9570% |
| Phase error | −1.92° | −2.12° | −0.8380° | −0.6428° |
| Cycle 5→6 waveform RMS difference | 0.0356% | 0.0360% | 0.034108% | 0.033814% |
| Corrected surface pressure, relative area/time L2 (sphere-exterior path) | 3.59% | 4.41% | 2.5300% | 2.2224% |
| Same with the two-sided path, outer side | 3.67% | 4.41% | — | — |
| Raw surface pressure, relative area/time L2 | 54.9% | 53.9% | 67.2% | 69.6% |
| Corrected pressure RMS / phase maximum, range over the five fields | 1.43–2.47% | 2.48–2.64% | 1.40–1.50% | 1.21–1.36% |
| Marker slip after a step (rms), largest over the run | 6.5e-4 | 3.7e-18 | — | — |

On this benchmark the K-matrix correction satisfies no-slip to round-off but is
not more accurate than the explicit update against the analytic solution: the
force error is 0.5 percentage points and the corrected pressure error 0.8
points larger at D16.  (IBFast's own table for D = 8h shows the same ordering
for the corrected surface pressure.)
The pressure correction works with either update: raw errors of 54–55% drop to
3.6–4.4%.

Cost of the two runs is not comparable from this pair: they shared the GPU with
each other and with a third run.  The K-matrix run made 6,720 constrained solves
with 778,155 CG iterations, 44 of them stopped by the stagnation guard and none
by the iteration limit; `K` (2,412 constraints) was rebuilt 1,680 times, which
took 1,350 s on the loaded host.  Both result directories pass `pyib-validate`.
Clean timings of the first 40 steps of this case are in
[PERFORMANCE.md](PERFORMANCE.md).
