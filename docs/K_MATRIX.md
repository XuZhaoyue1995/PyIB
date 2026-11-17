# K-matrix force correction (optional)

**Off by default.** With `ib.k_matrix.enabled = false` the solver runs the
explicit force update it always had.  With the previous form of the operator
(`solver.operator = "assembled"`) the numbers it writes are bit-identical to
those of the code before this option existed, on the bundled CPU examples that
were compared (see [PERFORMANCE.md](PERFORMANCE.md) and
`checks/upgrade-20261117.json`).

## What it does

The explicit update (`ib.force_mode = "accumulated_explicit"`) corrects the
marker force once per time step, from the slip left after the first inner
solve, and adds the correction to the force held from the previous steps.  The
no-slip condition is therefore met only through this step-to-step feedback.

With the K-matrix correction every inner iteration solves the momentum equation
and the no-slip condition together.  In the stream-function formulation of PyIB
the unknowns are the stream-function correction `dpsi` and one multiplier per
constraint, `lam`:

    A dpsi - B^T lam = b        b: edge residual with the force held so far
    B dpsi           = d        d: V_body - J u at the markers

`B = J · interp_f2c · curl` maps a stream-function correction to the velocity
change at the markers (`J` is the delta-kernel interpolation); `B^T` is the
spreading of a marker force taken to the edges.  The held marker force changes
by `lam / (dt ds^3)`.  The method is the constrained Picard iteration of IBFast
(`ib_update = 'pcg'`, `src/ibf_pcg.f90`):

1. `B` is assembled as a sparse matrix for the current marker positions and
   `K = B G^-1 B^T` is formed and factorised (`G^-1` is the diagonal
   preconditioner of the Helmholtz solve).  A fixed body needs this once; a
   moving body needs it every step.
2. Projected conjugate gradients with the exact constraint preconditioner
   `[G, B^T; B, 0]`: a feasible start `G^-1 B^T K^-1 d`, then CG on the momentum
   equation inside the subspace `B p = 0`.
3. One re-projection onto the constraint, the residual recomputed, and the
   multiplier `lam = -K^-1 B G^-1 r`.

After every inner iteration the marker velocities equal the body velocity to
round-off (for one constraint per marker).

## Switching it on

```json
"ib": {
  "force_mode": "accumulated_explicit",
  "k_matrix": {"enabled": true}
}
```

or `pyib-run case.json --outdir RUN --k-matrix on`.  A small bundled case:

```sh
pyib-run examples/cpu_oscillating_sphere_d8_k_matrix.json --outdir runs/k_matrix
pyib-validate runs/k_matrix
pyib-postprocess runs/k_matrix/final_field.npz --two-sided
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | the switch |
| `tolerance` | `1e-6` | relative tolerance of the projected residual |
| `max_iterations` | `5000` | iteration limit of one constrained solve |
| `xnorm_min` | `0.0` | lower bound of `\|psi\|^2` in the absolute part of the stopping test (see below) |
| `patch_size` | `1` | markers per constraint patch: 1 (every marker), 2 or 4 |
| `k_solver` | `"auto"` | `"inverse"` (explicit `K^-1`, one dense product per solve), `"cholesky"` (factor kept on the host) or `"auto"` (inverse while it fits) |

Requirements: `runtime.precision = "float64"`, `ib.force_mode =
"accumulated_explicit"`, `ib.implicit = false`.  Both `cpu` and `cuda` devices
are supported; `K` is always assembled and factorised on the host.

The stopping test is the one of the Helmholtz solve,
`r·g <= max(|psi|^2, xnorm_min) tol1^2 + tolerance^2 (r·g)_0` with
`tol1 = 1e-10`.  For a body moving through fluid at rest `|psi|^2` is small and
the test is much stricter than for the same flow seen from the body (uniform
stream); the IBFast record of this method puts the difference at two to three
orders of magnitude in the residual.  `xnorm_min` removes that dependence on the
frame: use `|psi|^2` of one step of the same case with the body at rest in a
uniform stream.  It is `0` by default (no change of the test).  The simpler way
to set it is `solver.cg_reference_speed = U`: PyIB then computes `|psi|^2` of a
uniform stream of speed `U` on the mesh and uses it as the floor of both the
explicit Helmholtz solve and the constrained solve.

The projected CG stops early when its residual has not improved for 100
iterations (IBFast's stagnation guard).  Such solves are counted in
`metadata.json` (`k_matrix.stagnations`); the history column
`helmholtz_tolerance_ratio` is then above 1 for that step.  A solve that reaches
`max_iterations` is an error when `solver.fail_on_cg_nonconvergence` is set.

## Constraint patches

With one marker per cell and a kernel three cells wide, neighbouring marker
constraints are nearly dependent; `K` then has a band of small eigenvalues and
its order and factorisation cost grow with the marker count.  With
`patch_size = 4` (or 2) the multiplier is constant on compact patches of markers
and no-slip is enforced for the patch sums; all markers still interpolate and
spread.  The patches are made on the reference surface by the rule of IBFast
`tools/make_ib_patches.py` (`ib_patches.make_patches`): rim markers of an open
surface are paired along the rim, interior markers are grouped by two pairwise
matchings.  Individual markers inside a patch may then slip; the patch sums do
not.

## What is written

`history.csv` keeps its columns.  With the correction on,
`ib_force_updates` equals the number of inner iterations,
`helmholtz_iterations` is the largest iteration count of the constrained solves
of the step.  `metadata.json` gains a `k_matrix` block: constraint and patch
counts, non-zeros of `B` and `K`, smallest and largest Cholesky pivot, number of
rebuilds and their time, number of solves, iterations, stagnations and
iteration-limit hits.

Field snapshots written with the correction on hold the momentum residual of
the finished step (`output.residual_state = "auto"` selects `"step_end"`), so the
pressure recovery sees the force that was actually applied.

## Checks (2026-11-17)

Unit tests (`tests/test_ib_constraint.py`): the sparse `B` equals the
interpolation of the curl and `B^T` equals the masked edge residual of the spread
force through the solver's own operators (1e-12); `K` is symmetric positive
definite and its solve is accurate to 1e-9; the projected CG solves a
manufactured saddle-point system; a fixed sphere has marker slip below 1e-10
after every step and builds `K` once; a moving sphere rebuilds `K` every step and
a restarted run reproduces the uninterrupted one bit for bit; with patches the
patch sums of the slip vanish; dependent constraints are reported as an error.

Against IBFast on the same discrete case (sphere oscillating in fluid at rest,
D = 8h, 201 markers, 560 steps = two periods; the IBFast run is the existing one
of 2026-11-16 made by its `tests/osc_sphere_pressure.sh` (`ib_update = 'pcg'`,
marker-by-marker constraint), not rerun here; PyIB on CUDA, float64, factored
operator, `cg_check_interval = 1`):

| | PyIB | IBFast |
|---|---|---|
| Force on the fluid, largest difference over the 560 steps / largest force | 3.4e-10 | (reference; `force.dat` has 12 digits) |
| Force of step 1 | 1.272169498150e-02 | 1.272169498150e-02 |
| CG iterations, all constrained solves | 126,943 | 126,851 |
| Solves stopped by the stagnation guard | 6 of 2,240 | 3 |
| Smallest / largest Cholesky pivot of `K` (603 constraints) | 3.67 / 14.4 | 3.67 / 14.4 |
| Marker slip after a step (rms) | below 4e-18 in every step | 1e-19 in the log lines inspected |

The explicit update of the two codes agrees to 2.7e-7 on the same case (IBFast
run of 2026-09-22).  The body force that includes the interior-fluid momentum
term differs by about 0.5% between the two codes with either update; the two
codes evaluate that term differently (analytic smoothed sphere indicator in
PyIB, triangulated interior in IBFast), and this difference is not new.

Cost on the same case (PyIB, CUDA, 560 steps): 340 s with the K-matrix correction
(126,943 CG iterations; `K` rebuilt every step, 21.7 s in total) against 440 s
with the explicit update.  PyIB does not record the total iteration count of the
explicit update; IBFast counts 309,433 for its explicit run of this case.

Against the analytic unsteady-Stokes solution (oscillating sphere, D = 16h, six
periods, [VALIDATION.md](VALIDATION.md)): body-force harmonic-vector error
12.15% with the K-matrix correction and 11.69% with the explicit update;
corrected surface pressure 4.41% and 3.59%.  The correction enforces no-slip
exactly and needs fewer CG iterations; on this smooth low-Reynolds benchmark it
is not more accurate than the explicit update.

Patches on a closed smooth body (fixed sphere, D = 8h, Re = 100, wake mesh,
120 steps to t = 3, CPU with the `threads` backend, 4 threads, on a machine that
was about 80% loaded by other work):

| Update | Constraints | Drag coefficient, mean of the last 20 steps | Against one constraint per marker | Largest marker slip after step 5 | Time |
|---|---:|---:|---:|---:|---:|
| Explicit | — | 1.2391 | −0.16% | 0.43 U (start-up transient) | 296 s |
| K-matrix, one per marker | 603 | 1.2412 | — | 3e-16 U | 220 s |
| K-matrix, patches of 2 | 318 | 1.2363 | −0.39% | 0.033 U | 204 s |
| K-matrix, patches of 4 | 180 | 1.2380 | −0.25% | 0.066 U | 212 s |

The flow has not reached a steady state at t = 3, so these are differences
between runs of the same length, not converged drag values.  On this small case
`K` is cheap either way; patches matter when the constraint count reaches the
thousands.

## Limits

- `K` is dense: memory `8 (3 N)^2` bytes for `N` constraints and a
  factorisation of `(3N)^3 / 3` operations — every step for a moving body.  One
  constraint per marker is practical up to a few thousand markers; beyond that
  use patches.
- Dependent constraints (markers much closer than one cell, two surfaces closer
  than the kernel width) make `K` singular; the run stops with an error.
- Only sphere cases can be run by the PyIB solver, so the correction was
  exercised on closed, smooth surfaces only.  The behaviour of patches on thin
  plates, studied with IBFast, was not reproduced here.
- The comparison with the analytic solution is in [VALIDATION.md](VALIDATION.md);
  the cost is in [PERFORMANCE.md](PERFORMANCE.md).
