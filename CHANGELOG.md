# Changelog

## 0.2.0 — 2026-11-22 (changes since `0.1.0rc2`, commit `a7003dd`)

First release after `0.1.0rc2`, tagged `v0.2.0`.  It also carries the CPU
backend work of 2026-10-05 (optional OpenMP CSR product, result schema 6),
which had not been released before; the changes here are built on top of it.
The solver source is that of commit `046b6f5`, with which the validation runs
below were made; the release adds documentation, check records, the version
number and the packaging change to it.

Numbers quoted here were measured on 2026-11-17 on one shared workstation
(i9-9900K, RTX 2080 8 GB, float64), single runs; the six-period validation
reruns and the six-period runs with the stopping floor followed on the same
machine until 2026-11-22.  Conditions and tables are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/VALIDATION.md](docs/VALIDATION.md).

### License

- PyIB is licensed under the BSD 3-Clause License from 2026-11-17 (`LICENSE`,
  `pyproject.toml`, `CITATION.cff`). The retained manuscript in `paper/` is
  excluded. `0.1.0rc2` as released carried no license.

### Packaging

- Version `0.2.0` (`pyproject.toml`, `pyib.__version__`, `CITATION.cff`).
- The manuscript PDF of `paper/` is no longer part of the source distribution
  or of the source archives of a release (`MANIFEST.in`, `.gitattributes`); it
  stays in the repository.

### Added

- **Two-sided surface-pressure correction** (`ib_pressure_surface.py`,
  `pyib-postprocess --two-sided [--far-gauge D] [--kernel h3|h3s|h4]
  [--surface FILE] [--raw-pressure mesh|lattice]`).  Port of IBFast
  `tools/pressure_lfc.py --two-sided`: triangulated surfaces, open or closed,
  both sides, faces dropped by geometry, consistent triangle winding, optional
  far-field pressure zero.  The existing sphere-exterior correction is unchanged
  and remains the default.  On an IBFast dump of an impulsively started
  zero-thickness disc the port and the IBFast tool agree to 4e-12 in the surface
  pressure.
- **K-matrix force correction** (`ib_constraint.py`, `ib.k_matrix.enabled`,
  `--k-matrix on`; **off by default**).  Port of the constrained Picard
  iteration of IBFast (`ib_update = 'pcg'`): momentum and no-slip solved
  together by projected CG with `K = B G^-1 B^T`.  Optional constraint patches
  (`ib.k_matrix.patch_size`, `ib_patches.py`, port of IBFast
  `tools/make_ib_patches.py`).  On the D8 oscillating sphere the force history
  of PyIB and of IBFast agree to 3.4e-10 over 560 steps.
  See [docs/K_MATRIX.md](docs/K_MATRIX.md).
- **Factored Helmholtz operator** (`solver.operator = "factored"`, new default;
  `"assembled"` keeps the previous single matrix) and in-place CG vector
  updates.  D16 benchmark case: 2.17 times faster on CUDA, 3.6 times on CPU;
  the D32 case now fits an 8 GB card.  Same CG iteration counts; forces differ
  from the previous code by 1e-13 or less (relative) on the compared runs.
- **`threads` CPU backend** (`solver.cpu_sparse_backend = "threads"`): SciPy's
  CSR kernel on a thread pool, bit-identical to SciPy, no compiled extension.
- **Optional floor of the CG stopping test** (`solver.cg_reference_speed`, off
  by default) for bodies moving through fluid at rest: over the first 40 steps
  of the D16 case 3.9 times fewer CG iterations and 3.6 times less time, at a
  force change of 3.6e-7.  Over the six periods of the benchmark the runs take
  1.7 (D16), 2.9 (D32) and 2.4 (D40) times less time; the force errors are
  unchanged at four decimals and the corrected pressure errors change by at
  most 1.3e-4 percentage points.
- **Step-end residual** (`output.residual_state`): field snapshots can keep the
  momentum residual of the finished step (default with the K-matrix correction)
  instead of that of the last inner iteration (default otherwise, as before).
- A bundled K-matrix example, `examples/cpu_oscillating_sphere_d8_k_matrix.json`.
- Tests: `tests/test_ib_pressure_surface.py`, `tests/test_ib_constraint.py`,
  `tests/test_ib_operator.py` (66 tests in total on Windows with the OpenMP class skipped; 71 on Linux
  with the OpenMP extension built, none skipped).
- Documents: `docs/K_MATRIX.md`, `docs/PERFORMANCE.md`, this file.

### Changed

- Result schema 7, checkpoint schema 5: the configuration stored with a run has
  the new keys (`ib.k_matrix`, `solver.operator`, `solver.cg_reference_speed`,
  `output.residual_state`).  Results and checkpoints of earlier schemas stay
  usable with the release that wrote them; they are not migrated.
- `pyib-validate` accepts the new keys and K-matrix runs.  For those the
  per-step force-update count equals the number of inner iterations and a
  projected-CG solve that stops on its stagnation guard is accepted.

### Unchanged, and how that was checked

- With `solver.operator = "assembled"` and the K-matrix correction off, the new
  code gives bit-identical `history.csv` columns and final fields to the
  previous code on the three bundled CPU examples that were compared.
- The sphere-exterior pressure correction (`pyib-postprocess` without
  `--two-sided`) has the same code path as before; its tests pass unchanged.
- The six-period D32 and D40 validation runs were repeated in full with the new
  code (default factored operator, CUDA, an 8 GB workstation GPU) and reproduce
  the archived figures: force harmonic-vector error 5.2690% / 4.1190%,
  corrected surface-pressure error 2.5300% / 2.2224%; body force within
  6.7e-11 / 1.1e-10 of the archived histories over all sampled steps.  Both
  reruns were resumed from checkpoints after interruptions.  See
  [docs/VALIDATION.md](docs/VALIDATION.md) and
  `checks/validation-reruns-20261122.json`.

### Not done

- The standard runner (`pyib-run`) and its load correction are still set up
  for sphere cases only.  This is a limit of the bundled cases, not of the
  method: the solver core takes any set of surface markers with a prescribed
  motion, but no other geometry is bundled, tested or validated in this
  release.  Open surfaces reach the two-sided pressure correction through
  snapshots that carry their own markers and triangles (for example converted
  IBFast dumps).
- No smoothed kernel (`h3s`) in the solver; the pressure tool supports it for
  such snapshots.
- No wall-shear recovery.
- The K-matrix correction was run over six periods at D16 only, not at D32 or
  D40; see docs/VALIDATION.md.
- The D40 case does not fit an 8 GB card with CuPy's default memory pool.  The
  D40 reruns used a capped pool with fallback to direct allocations set up by
  the launching script; PyIB itself has no such option.
- `K` of the K-matrix correction is rebuilt on the host every step for a moving
  body (0.64 s per step at D16); this was not optimised.
