# PyIB solver and pressure release checks

## Package identity

The package and import namespace are `pyib`, version `0.1.0rc2`.
The public commands are `pyib-run`, `pyib-postprocess` and `pyib-validate`.
The sole software author is **Zhaoyue Xu**; the license remains pending.

The 2026-09-22 packaging check built and non-editably installed PyIB,
verified all three command help pages, and completed the fixed-sphere example
for five CPU/NumPy float64 steps. The result validator found full-field
coverage and five history rows. Its force-balance error was `6.077416e-16`
against a `1e-9` threshold. Installation metadata is in
[pyib-installation.json](pyib-installation.json).

## Current source and postprocessing checks — 2026-09-25

This release contains the CPU/CUDA solver, pressure-field reconstruction,
and surface-pressure correction. Runtime body-force accounting remains part
of the solver. Surface-pressure force is reported separately from total force.

```sh
python tools/verify_source_manifest.py
python tools/run_tests.py
python tools/check_pressure_validation.py
```

The manifest records 13 unchanged captured source files and the updated
postprocessor. Its original source hash is retained separately for provenance.
The time integration, pressure algorithms and retained manuscript are unchanged.
Installation and integration results for this release are recorded in
[pressure-release-checks.json](pressure-release-checks.json).

The independent evidence checker reproduces the pressure and runtime-force
benchmark metrics from the bundled D32/D40 measurements and analytical
reference. See [validation](../docs/VALIDATION.md) for the physical problem,
error definitions and refinement comparison, and
[pressure-validation-20260925.json](pressure-validation-20260925.json) for
machine-readable evidence.
