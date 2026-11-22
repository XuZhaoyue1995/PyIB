# Check records

Records of what was run and measured for each state of the source, newest
first.  They are evidence kept with the code, not documentation of how to use
it; the documents in [`docs/`](../docs) explain the methods and summarise the
numbers.

| File | Date | What it records |
|---|---|---|
| [release-0.2.0-checks.json](release-0.2.0-checks.json) | 2026-11-22 | Build, installation, example-run and test checks of the `0.2.0` release. |
| [validation-reruns-20261122.json](validation-reruns-20261122.json) | 2026-11-22 | Six-period runs with the source of commit `046b6f5`: full reruns of the archived D32 and D40 cases (metrics, comparison with the archived histories, restarts, timings) and the benchmark at D16, D32 and D40 with the optional floor of the CG stopping test against the runs without it. Summarised in [VALIDATION](../docs/VALIDATION.md) and [PERFORMANCE](../docs/PERFORMANCE.md). |
| [upgrade-20261117.json](upgrade-20261117.json) | 2026-11-17 | Changes after `0.1.0rc2` (factored operator, K-matrix correction, two-sided pressure correction): equivalence with the previous code, timings, comparison with IBFast, D16 six-period runs, first steps of D32 against the archived history. Summarised in [PERFORMANCE](../docs/PERFORMANCE.md), [K_MATRIX](../docs/K_MATRIX.md) and [VALIDATION](../docs/VALIDATION.md). |
| [cpu-csr-regression-20261005.json](cpu-csr-regression-20261005.json) | 2026-10-05 | Optional OpenMP CSR backend: unit tests and short fixed/oscillating-sphere runs on a Linux compute node. Summarised in [CPU_ACCELERATION](../docs/CPU_ACCELERATION.md). |
| [cpu-csr-comparison-20261005.json](cpu-csr-comparison-20261005.json) | 2026-10-05 | Member-by-member comparison of the outputs of those runs (SciPy against OpenMP, restart against uninterrupted). |
| [pressure-validation-20260925.json](pressure-validation-20260925.json) | 2026-09-25 | Summary of the D32/D40 oscillating-sphere validation of `0.1.0rc2` ([VALIDATION](../docs/VALIDATION.md)). |
| [pressure-validation-20260925.zip](pressure-validation-20260925.zip) | 2026-09-25 | The evidence itself: cases, histories, per-marker surface pressures and a standalone checker. `python tools/check_pressure_validation.py` recomputes the published figures from it. |
| [pressure-release-checks.json](pressure-release-checks.json) | 2026-09-25 | Build, installation and command-line checks of the `0.1.0rc2` release. |
| [pyib-packaging-checks.md](pyib-packaging-checks.md) | 2026-09-25 | Narrative of the `0.1.0rc2` packaging and source-identity checks. Its statement that the license is pending describes that date; the license was chosen on 2026-11-17. |
| [pyib-installation.json](pyib-installation.json) | 2026-09-22 | Installation metadata of the first packaged build (`0.1.0rc1`). |
| [packaging-checks.md](packaging-checks.md) | 2026-09-22 | Initial packaging check, made before the package was named PyIB. |
