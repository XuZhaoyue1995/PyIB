# Initial packaging checks — 2026-09-22

The initial distribution was installed as a non-editable wheel under the
temporary name `cfdagent-ib==0.1.0rc1`. Build dependencies were isolated;
runtime NumPy/SciPy/CuPy dependencies were unchanged. All three console
commands returned help successfully, and the fixed-sphere CPU example
completed five steps with valid result structure.

The public package was subsequently named PyIB. Current source identity,
postprocessing checks and validation are recorded in
[pyib-packaging-checks.md](pyib-packaging-checks.md) and
[release status](../RELEASE_STATUS.md).
