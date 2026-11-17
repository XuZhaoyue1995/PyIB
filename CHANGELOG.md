# Changelog

## Unreleased — working tree of 2026-11-17 (on top of `0.1.0rc2`, commit `a7003dd`)

Nothing below has been tagged or released; the package version is still
`0.1.0rc2`.  The same commit also carries the CPU backend work of 2026-10-05
(optional OpenMP CSR product, result schema 6), which had not been committed
before; the changes here are built on top of it.

Numbers quoted here were measured on 2026-11-17 on one shared workstation
(i9-9900K, RTX 2080 8 GB, float64), single runs; conditions and tables are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/VALIDATION.md](docs/VALIDATION.md).

### License

- PyIB is licensed under the BSD 3-Clause License from 2026-11-17 (`LICENSE`,
  `pyproject.toml`, `CITATION.cff`). The retained manuscript in `paper/` is
  excluded. `0.1.0rc2` as released carried no license.

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
  by default) for bodies moving through fluid at rest: on the D16 case 3.9 times
  fewer CG iterations and 3.6 times less time, at a force change of 3.6e-7.
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

### Not done

- The PyIB solver still runs sphere cases only; open surfaces reach the
  two-sided pressure correction through snapshots that carry their own markers
  and triangles (for example converted IBFast dumps).
- No smoothed kernel (`h3s`) in the solver; the pressure tool supports it for
  such snapshots.
- No wall-shear recovery.
- The full six-period D32 and D40 validation runs were not repeated (about 12
  hours for D32 alone on the available 8 GB card).  Rerun instead: the first 10
  steps of D32 against the archived history (body force within 3.5e-14, same
  iteration counts) and the benchmark at D16 over six periods with both force
  updates; see docs/VALIDATION.md.
- `K` of the K-matrix correction is rebuilt on the host every step for a moving
  body (0.64 s per step at D16); this was not optimised.
