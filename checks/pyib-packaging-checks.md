# PyIB packaging and integration checks, 2026-09-22

These are fresh checks of `pyib==0.1.0rc1` after the author selected the name
**PyIB — A Python Immersed Boundary Solver for CPUs and GPUs**. Earlier checks
under the temporary name remain unchanged in `packaging-checks.md`.

## Scope of the rename

- Moved the local candidate from `release/cfdagent-ib` to `release/PyIB` and
  its Python namespace from `src/cfdagent_ib` to `src/pyib` after verifying
  that both absolute move targets were inside the workspace release directory.
- Updated packaging metadata, documentation, wrappers, testing tools,
  attributes and manifest destination paths. Original source paths, capture
  time, source sizes and SHA-256 values remain unchanged.
- Verified all 14 numerical/source files against both the original tree and
  the installed package: byte-for-byte identity is preserved. The original
  source tree was not edited.
- Retained the internal historical `solver/cfdagent_ib.py` filename and log
  prefix. Public commands are `pyib-run`, `pyib-postprocess`, `pyib-validate`.
- Archived obsolete generated build/egg-info directories inside ignored
  `.packaging-history/`, also explicitly pruned from source distributions. Fresh
  `build/lib` contains only `pyib`.

## Installation and checks

- Uninstalled `cfdagent-ib` from the workspace virtual environment.
- Built and non-editably installed `pyib==0.1.0rc1` with `pip install --no-deps`;
  existing runtime dependencies were not changed.
- Import resolves to `E:\CFDAgentV\.venv\Lib\site-packages\pyib\__init__.py`.
  `direct_url.json` points to the local `release/PyIB` candidate with no
  editable flag. The old top-level `cfdagent_ib` namespace and old command
  executables are absent. Details: `pyib-installation.json`.
- All three installed console commands returned `--help` with exit code 0.
- `python tools/verify_source_manifest.py` verified all 14 files.
- `python tools/run_tests.py` passed all 12 preserved offline postprocess
  unittest cases against the installed package in 0.414 s.
- Installed `pyib-run` completed the unchanged fixed-sphere example for five
  CPU/NumPy float64 steps in `runs/pyib-cpu-smoke-20260922`.
- Installed `pyib-validate` returned `VALID`, full-field coverage and five
  history rows. Reported and independently recomputed force-balance error:
  `6.077416e-16` against a `1e-9` threshold.
- Installed `pyib-postprocess` completed full pressure-field reconstruction,
  pressure-band correction and wall-shear recovery with `--vtk` on the existing
  4,096-cell audit snapshot. Fresh outputs are in
  `runs/pyib-postprocess-20260922`: NPZ, surface CSV, diagnostics JSON and two
  VTU point-cloud files.

These are packaging/integration and manufactured-field checks, not a physical
accuracy certificate. The extremely coarse five-step audit field still has
about 64.87% surface-load/volume-IB load discrepancy. D32/D40 physical evidence
remains pending. CPU and CUDA paths are included; no new GPU run or distributed
execution test was performed for this rename. License and public repository
destination remain undecided. Nothing was uploaded to GitHub or a registry.
