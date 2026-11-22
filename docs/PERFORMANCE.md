# Performance: where the time goes and what was changed

Measured on 2026-11-17 on one workstation: Intel i9-9900K (8 cores), NVIDIA RTX
2080 (8 GB), Windows 11, Python 3.12.5, NumPy 2.5.3, SciPy 1.18.1, CuPy 13.6.0,
float64.  The machine was shared: other work kept the CPU at 22–56% load during
the measurements.  Every number is a single run, not an average.  They describe
this case on this machine and are not a general speed guarantee.

Case: the oscillating-sphere benchmark set-up of [VALIDATION.md](VALIDATION.md)
at 16 cells per diameter (147,456 cells, 467,232 edges, 804 markers, time step
T/280, four inner iterations, `cg_check_interval = 20`), from the start of the
run.  Times are the sums of the per-step times in `history.csv`; mesh
construction and operator assembly are not included.  "As found" is the source
before this work (commit `a7003dd` plus the uncommitted CPU-backend changes of
2026-10-05).

## Profile of the code as found

| | CPU | CUDA |
|---|---:|---:|
| Share of the step in the Helmholtz CG | 99.1% | 97.0% |
| Sparse product `A @ p` | 88.7% of the step, 74 ms each | 60.1%, 2.1 ms each |
| Rest of one CG iteration (vector updates, dot products) | 9 ms | 1.3 ms |
| Everything outside the CG (velocity, convection/diffusion, IB) | < 1% | < 3% |

The assembled operator `A = R I V P C` has 119.8 non-zeros per edge row
(56 million at D16).  At 32 cells per diameter that is about 436 million
non-zeros, 5.2 GB as a CSR matrix: on the 8 GB card the first step of the D32
case did not finish in 5.4 minutes and 7.4 GB had spilled into shared system
memory.

## Changes

1. **Factored operator** (`solver.operator = "factored"`, the new default).  The
   same operator is applied as three sparse factors,
   `A p = (R I) (V ((P C) p))`, with about 23 non-zeros per edge row in total
   instead of 120.  `"assembled"` keeps the single product matrix.
2. **In-place CG updates.**  The vector updates of one CG iteration are done
   without temporaries (NumPy `out=`; on CUDA two fused kernels with
   multiply-add contraction switched off).  The multiplies and adds per element
   are the same as before.
3. **`threads` CPU backend** (`solver.cpu_sparse_backend = "threads"`).  SciPy's
   CSR kernel applied to row blocks on a thread pool; no compiled extension, any
   platform, bit-identical to SciPy.  See [CPU_ACCELERATION.md](CPU_ACCELERATION.md).
4. **Optional floor of the CG stopping test** (`solver.cg_reference_speed`, off
   by default).  The stopping test of the Helmholtz solve is
   `r·z < |psi|^2 tol1^2 + tol2^2 (r·z)_0`.  For a body moving through fluid at
   rest `|psi|^2` is tiny, the absolute part vanishes, and the solve is carried
   much further than for the same flow seen from the body (the IBFast record of
   its constrained solve describes the same effect).  With
   `cg_reference_speed = U` the term `|psi|^2` is bounded below by
   its value for a uniform stream of speed `U` on the same mesh.  This changes
   the accepted iterate, so it changes the results at the level shown below
   (first 40 steps, and whole six-period runs at D16, D32 and D40); it is a
   trade the user chooses, not a default.

## Measured times

CUDA, D16, first 40 steps:

| Configuration | Time | Speed-up | Body force against "as found" (max over the 40 steps, relative to the largest force) |
|---|---:|---:|---|
| As found | 176.5 s | 1.00 | — (two runs of the as-found code differ by 2.5e-14 over 12 steps: CUDA runs are not bit-reproducible) |
| New, `operator = "assembled"` | 169.5 s | 1.04 | 8.4e-14 |
| New, `operator = "factored"` (default) | 81.5 s | 2.17 | 2.5e-13; same CG iteration counts in every step |
| New, factored, `cg_reference_speed = 0.02` (peak body speed) | 22.4 s | 7.9 | 3.6e-7; 6,040 instead of 23,740 iterations (sum over the steps of the largest inner solve) |
| New, factored, K-matrix correction on | 80.7 s | — | different method (see [K_MATRIX.md](K_MATRIX.md)); 25.6 s of it is rebuilding `K` every step |
| New, factored, K-matrix on, `cg_reference_speed = 0.02` | 43.2 s | — | 3.0e-6 against the K-matrix run without the floor |

CPU, D16, first 3 steps:

| Configuration | Time | Speed-up |
|---|---:|---:|
| As found (SciPy, assembled) | 450.0 s | 1.00 |
| New, factored, SciPy | 125.2 s | 3.6 |
| New, factored, `threads` backend, 8 threads | 100.5 s | 4.5 |
| New, assembled, `threads` backend, 8 threads | 238.4 s | 1.9 |

After the change one CPU CG iteration takes 25 ms instead of 83 ms; the three
factor products are 19.5 ms of it.

CUDA, D32 (1,179,648 cells, 3,637,824 edges): the as-found code could not run
on this card (see above).  With the factored operator the first 12 steps took
468 s (22–75 s per step during the start-up, where the inner solves need up to
2,220 iterations).

## The stopping floor over whole runs

Measured from 2026-11-17 to 2026-11-22 on the same workstation: the six-period
benchmark runs of [VALIDATION.md](VALIDATION.md) with the factored operator on
CUDA, without and with `cg_reference_speed = 0.02`.  Times are again sums of
the per-step times in `history.csv`.

| | D16 (1,680 steps) | D32 (3,360 steps) | D40 (4,200 steps) |
|---|---:|---:|---:|
| Without the floor | 2,135 s (1.27 s per step) | 43,698 s (13.0 s per step) | 100,142 s (23.8 s per step) |
| With the floor | 1,254 s (0.75 s per step) | 13,502 s (4.02 s per step) | 37,442 s (8.91 s per step) |
| Speed-up, total time | 1.70 | 3.24 | 2.67 |
| Speed-up, median time per step | 1.83 | 2.89 | 2.44 |
| Speed-up, first 10 steps | 2.87 | 6.60 | 6.28 |
| Iterations of the largest inner solve, sum over the sampled steps | 151,300 → 47,240 | 439,120 → 183,900 | 603,600 → 283,020 |
| Body force against the run without the floor (largest difference over the run, relative to the largest force) | 5.6e-7 | 4.0e-6 | 6.9e-6 |

- The floor saves most during the start-up, where the inner solves need most
  iterations; the 40-step figure of the table above (81.5 s against 22.4 s) is
  such a start-up figure.  Over a whole run the gain is 1.7 at D16 and between
  2.4 and 3.2 at D32 and D40.
- The two D16 runs were made on the same day, each alone on the GPU.  The D32
  and D40 runs without the floor are the validation reruns; they ran in several
  legs while the workstation was in other use and contain slow stretches, so
  for them the ratio of the median times per step is the safer figure.
- D40 does not fit the 8 GB card with CuPy's default memory pool.  Both D40
  runs used a capped pool with fallback to direct allocations, set up by the
  launching script and not part of PyIB.  Whether that is why the gain at D40
  is smaller than at D32 was not measured.
- The error figures of the benchmark change by at most 1.3e-4 percentage points
  with the floor (table in [VALIDATION.md](VALIDATION.md)).

The runs are recorded in `checks/validation-reruns-20261122.json`.

## Results unchanged?

- **`operator = "assembled"`, K-matrix off, CPU:** bit-identical to the as-found
  code on the three bundled examples that were compared
  (`cpu_fixed_sphere_smoke`, `cpu_oscillating_sphere_smoke`,
  `cpu_fixed_sphere_re100_d8_gate`): every `history.csv` column except the
  timing, and `s`, `vc`, `ib_force_L`, `ib_force_E`, `momentum_residual`,
  `body_force` of `final_field.npz`.
- **`operator = "factored"` (default), CPU, same examples:** same CG iteration
  counts; body force differs by 1.6e-16 to 6.3e-13 and the final velocity field
  by at most 1.8e-10, relative to their largest values.
- **CUDA:** see the table; the differences are at the level of the run-to-run
  differences of the as-found code itself.

The comparisons are recorded in `checks/upgrade-20261117.json`.

## What was not done

- No better preconditioner: the CG still uses the diagonal one and needs several
  hundred to a few thousand iterations per solve on these meshes.
- The remaining CUDA cost at D32 is memory traffic of the factor products and
  vector updates on an 8 GB consumer card; no structured (matrix-free) kernel
  was written.
- `K` of the K-matrix correction is assembled, factorised and inverted on the
  host every step for a moving body (0.64 s per step at D16 with 2,412
  constraints).
- The `openmp` backend (Linux only) was rebuilt and checked for bit-identical
  results on WSL, not timed.
