# Solver and pressure validation: D32/D40 oscillating sphere

Author: **Zhaoyue Xu**. Results retrieved and independently reviewed on
2026-09-25. The D32 and D40 simulations completed by 2026-09-23. Both were
repeated in full with the changed source in November 2026 and reproduced the
figures below ([reruns](#full-six-period-reruns-of-d32-and-d40)).

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

## Checks of the changes after `0.1.0rc2` (2026-11-17 to 2026-11-22, not yet released)

The results above were produced by the released `0.1.0rc2` code on a cluster
GPU.  The source has since changed how the Helmholtz operator is applied (three
factors instead of one assembled matrix, in-place CG updates) and adds an
optional K-matrix force correction, a two-sided pressure correction and an
optional floor of the CG stopping test ([CHANGELOG.md](../CHANGELOG.md)).  Both
archived cases were run again in full with the changed code, and the benchmark
was also run with the optional floor.  Everything in this section was run on
one workstation (RTX 2080, 8 GB, CuPy 13.6) in float64, single runs.  The
machine-readable records are
[checks/upgrade-20261117.json](../checks/upgrade-20261117.json) and
[checks/validation-reruns-20261122.json](../checks/validation-reruns-20261122.json).

### Full six-period reruns of D32 and D40

The D32 and D40 cases of the evidence archive were run from t = 0 to six
periods with the source of commit `046b6f5` (default factored operator, CUDA)
and evaluated with the definitions of the tables above.  The archived runs used
the single assembled operator, CuPy 14.1 and another GPU.

| Quantity | D32 archived | D32 rerun | D40 archived | D40 rerun |
|---|---:|---:|---:|---:|
| Runtime body-force harmonic-vector error | 5.2690% | 5.2690% | 4.1190% | 4.1190% |
| Runtime body-force amplitude error | +5.0513% | +5.0513% | +3.9570% | +3.9570% |
| Phase error | −0.8380° | −0.8380° | −0.6428° | −0.6428° |
| Cycle 5→6 waveform RMS difference | 0.034108% | 0.034108% | 0.033814% | 0.033814% |
| Corrected surface-pressure relative area/time L2 error | 2.5300% | 2.5300% | 2.2224% | 2.2224% |
| Raw surface-pressure relative area/time L2 error | 67.2063% | 67.2063% | 69.6260% | 69.6260% |
| Body force (x) at the sampled steps, largest difference from the archived history / largest force | | 6.7e-11 (675 steps) | | 1.1e-10 (843 steps) |
| CG iterations of the largest inner solve, sampled steps with the archived count | | 674 of 675 | | 843 of 843 |

The fitted force coefficients agree with the archived ones to 1.0e-13 (D32) and
4.9e-13 (D40) relative, the corrected pressure errors to 1e-10 relative.  Both
result directories pass `pyib-validate`.

Neither run was uninterrupted.  D32 was paused once by hand and interrupted
once by a restart of the computer; it was resumed from the checkpoints at steps
560 and 1,680.  D40 was interrupted at step 1,155 by a restart of the computer
and resumed from the checkpoint at step 700.  The agreement above therefore
includes restarts from checkpoints.  Stepping took 43,698 s for D32 (13.0 s per
step) and 100,142 s for D40 (23.8 s per step) on a workstation that was in
other use during the runs.  D32 fits the 8 GB card.  D40 does not fit it with
CuPy's default memory pool; the D40 runs of this section used a capped pool
with fallback to direct allocations, set up by the launching script, which is
not part of PyIB.

A first comparison on 2026-11-17 covered only steps 1–10 of D32 (body force
within 3.5e-14 of the archived history, same iteration counts); it is kept in
`upgrade-20261117.json`.

### Optional floor of the CG stopping test over six periods

`solver.cg_reference_speed` is off by default.  When set it changes the iterate
at which the Helmholtz CG stops ([PERFORMANCE.md](PERFORMANCE.md)), so it
changes the results slightly.  The benchmark was run over six periods with
`cg_reference_speed = 0.02` (the peak body speed) and otherwise identical
cases, and compared with the runs without the floor: at D16 the explicit-update
run of the last table of this document, at D32 and D40 the reruns above.
Entries read "without floor → with floor".  The time and iteration counts of the
D16 run without floor are from a repeat of that run on 2026-11-21, alone on
the GPU like the run with the floor; the repeat follows the first run to
2.1e-11 in the body force.

| Quantity | D16 | D32 | D40 |
|---|---:|---:|---:|
| Runtime body-force harmonic-vector error | 11.6866% → 11.6866% | 5.2690% → 5.2690% | 4.1190% → 4.1190% |
| Change of that error, percentage points | +1.0e-5 | +1.8e-5 | +2.0e-5 |
| Corrected surface pressure, relative area/time L2 (sphere-exterior path) | 3.5912% → 3.5912% | 2.5300% → 2.5299% | 2.2224% → 2.2223% |
| Same with the two-sided path, outer side | 3.6675% → 3.6675% | 2.4164% → 2.4163% | 2.2248% → 2.2246% |
| Change of the corrected pressure error (sphere-exterior path), percentage points | −1.2e-5 | −1.3e-4 | −1.3e-4 |
| Corrected pressure RMS of a field (sphere-exterior path), relative change, range over the five fields | −5.0e-6 to +3.7e-6 | −5.7e-5 to −4.8e-5 | −7.3e-5 to −4.4e-5 |
| Body force (x) at the sampled steps, largest difference / largest force | 5.6e-7 | 4.0e-6 | 6.9e-6 |
| CG iterations of the largest inner solve, sum over the sampled steps | 151,300 → 47,240 | 439,120 → 183,900 | 603,600 → 283,020 |
| Stepping time | 2,135 s → 1,254 s | 43,698 s → 13,502 s | 100,142 s → 37,442 s |

With the floor the force errors are unchanged at the four decimals shown and
the corrected pressure errors change by at most 1.3e-4 percentage points.  The
D40 two-sided entry (2.2248% → 2.2246%) is the same change of 1.3e-4 points
crossing two rounding boundaries.  The change is ten times larger at D32 than
at D16 and did not grow further from D32 to D40.  All three runs with the floor
pass `pyib-validate`.  The floor stays off by default and the figures of this
document are those without it.  Timing conditions and speed-ups are discussed
in [PERFORMANCE.md](PERFORMANCE.md).

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
figures of the released code are repeated for the trend (the two-sided values
are from the field snapshots of the reruns above).

| Quantity | D16 explicit | D16 K-matrix | D32 (archived) | D40 (archived) |
|---|---:|---:|---:|---:|
| Runtime body-force harmonic-vector error | 11.69% | 12.15% | 5.2690% | 4.1190% |
| Runtime body-force amplitude error | +11.14% | +11.51% | +5.0513% | +3.9570% |
| Phase error | −1.92° | −2.12° | −0.8380° | −0.6428° |
| Cycle 5→6 waveform RMS difference | 0.0356% | 0.0360% | 0.034108% | 0.033814% |
| Corrected surface pressure, relative area/time L2 (sphere-exterior path) | 3.59% | 4.41% | 2.5300% | 2.2224% |
| Same with the two-sided path, outer side | 3.67% | 4.41% | 2.4164% (rerun) | 2.2248% (rerun) |
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
