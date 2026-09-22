# Historical checks under the temporary package name, 2026-09-22

- Installed a non-editable wheel as `cfdagent-ib==0.1.0rc1` in the existing
  project virtual environment, with `pip install --no-deps`.
- Build dependencies were provisioned in pip's isolated temporary build
  environment. Runtime NumPy/SciPy/CuPy dependencies were not changed.
- All three installed console commands returned help successfully.
- The 14 packaged source files match the original files byte-for-byte and
  pass `tools/verify_source_manifest.py`.
- `python tools/run_tests.py` ran 12 preserved offline postprocess unittest
  cases against the installed package: all passed in 0.318 s.
- No production simulation or GPU validation is established by these checks.
  D32/D40 physical evidence remains pending.

Independent follow-up against the installed package also completed:

- `cfdagent-ib-run` ran the supplied fixed-sphere example for 5 CPU steps
  into a new workspace run directory.
- `cfdagent-ib-validate` reported `VALID` for that installed-package run.
- `cfdagent-ib-postprocess` completed all three offline stages on the
  4,096-cell audit snapshot using default correction settings.
- These remain integration checks. In particular, the coarse audit
  snapshot is not a physical accuracy benchmark.

This report contains historical packaging evidence under the temporary name;
it has not been relabelled as PyIB testing. The author subsequently selected
the public name PyIB. License, repository destination and source cleanup
remain pending as described in `RELEASE_STATUS.md`. Fresh PyIB test results
are recorded separately in `pyib-packaging-checks.md`.
