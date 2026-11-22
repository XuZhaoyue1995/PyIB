# PyIB — A Python Immersed Boundary Solver for CPUs and GPUs

[简体中文](README.zh-CN.md)

PyIB is an independent Python immersed-boundary solver with NumPy/SciPy CPU
and CuPy CUDA backends. It provides JSON case configuration, automatic mesh
construction, checkpoint/restart, result checks, and offline pressure-field
reconstruction and surface-pressure correction. CFDAgent can call PyIB through its command-line
interface.

**The solver and pressure modules have completed numerical validation on the
D32/D40 oscillating-sphere benchmark (Re₀ = 0.2).** At D40, the runtime
force harmonic-vector error is **4.1190%** and corrected surface-pressure
relative area/time L2 error is **2.2224%**. The validation covers analytic
comparison, periodic convergence, joint grid/time refinement and source
provenance. See the [complete validation record](docs/VALIDATION.md).

**Sole software author: Zhaoyue Xu.** Software attribution is recorded in
[AUTHORS.md](AUTHORS.md) and [CITATION.cff](CITATION.cff). The Python distribution
and import namespace are `pyib`; the current release is `0.2.0`.

PyIB is licensed under the [BSD 3-Clause License](LICENSE). The retained
manuscript in `paper/` is not covered by that license
([note](paper/README.md)). If you use PyIB in published work, please cite it
using [CITATION.cff](CITATION.cff). See also [release status](RELEASE_STATUS.md).

**New in `0.2.0` (2026-11-22), since `0.1.0rc2`:** a faster
Helmholtz operator (2.2 times on CUDA and 3.6 times on CPU on the D16 benchmark
case, measured), an optional K-matrix force correction (off by default) and a
two-sided surface-pressure correction for triangulated surfaces, both ported
from IBFast. The six-period D32/D40 validation runs were repeated in full with
these changes and reproduced the archived errors. See the
[changelog](CHANGELOG.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/pyib/` | The package: mesh generator and `solver/` (time stepping, immersed boundary, K-matrix correction, pressure postprocessing, result validator) |
| `examples/` | Small JSON cases for installation and execution checks |
| `tests/` | Unit tests (`python tools/run_tests.py`) |
| `tools/` | Source-manifest check, test runner, validation-evidence checker |
| `docs/` | Method and measurement documents, listed below |
| `checks/` | Dated records of what was run and measured ([index](checks/README.md)) |
| `paper/` | The retained original manuscript draft |

Documents: [validation](docs/VALIDATION.md) ·
[pressure postprocessing](docs/POSTPROCESSING.md) ·
[K-matrix force correction](docs/K_MATRIX.md) ·
[performance](docs/PERFORMANCE.md) ·
[CPU acceleration](docs/CPU_ACCELERATION.md) ·
[dependencies](docs/DEPENDENCIES.md) · [changelog](CHANGELOG.md) ·
[release status](RELEASE_STATUS.md)

## CPU installation and first run

Use Python 3.10 or newer in a virtual environment, from this directory:

```sh
python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed
pyib-validate runs/fixed
```

The bundled JSON examples include fixed and oscillating spheres, plus
fixed-sphere Re=100 D8/D16 configurations and one small case with the K-matrix
force correction switched on (`cpu_oscillating_sphere_d8_k_matrix.json`).

### Optional parallel CPU sparse multiplication

On Linux with a GCC-compatible OpenMP compiler, build the optional extension
at installation time:

```sh
PYIB_BUILD_OPENMP=1 python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed-openmp --cpu-sparse-backend openmp --cpu-sparse-threads 4
```

Ordinary installations continue to use SciPy. The extension parallelizes CSR
matrix rows for CPU float64 calculations; it does not change the flow model,
CG algorithm, or GPU kernels. See [CPU acceleration](docs/CPU_ACCELERATION.md)
for configuration, tested scope, and restart compatibility.
The source regression passed 23 unit tests and fixed/moving multi-step and
restart comparisons, with identical compared numerical states.

### Threads without a compiler

```sh
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed-threads --cpu-sparse-backend threads --cpu-sparse-threads 8
```

The `threads` backend applies SciPy's CSR kernel to row blocks on a thread
pool; it needs no extension, runs on any platform and reproduces the SciPy
result bit for bit.

## Solver options added in 0.2.0

| Option | Default | Effect |
|---|---|---|
| `solver.operator` (`--operator`) | `"factored"` | Helmholtz operator as three sparse factors (about 23 non-zeros per edge instead of 120): same operator up to round-off, a fifth of the memory. `"assembled"` reproduces earlier results bit for bit on CPU. |
| `solver.cg_reference_speed` (`--cg-reference-speed`) | `0` (off) | Frame-independent floor of the CG stopping test for bodies moving through fluid at rest; fewer iterations (benchmark runs 1.7 to 2.9 times faster); over six periods the benchmark body force changes by 6e-7 (D16) to 7e-6 (D40) of its largest value. |
| `ib.k_matrix.enabled` (`--k-matrix on`) | `false` | Momentum and no-slip solved together in every inner iteration (IBFast `ib_update='pcg'`). |
| `output.residual_state` | `"auto"` | Which momentum residual a field snapshot keeps: of the last inner iteration (explicit update) or of the finished step (K-matrix). |

Measured times and the comparison with the previous results are in
[performance](docs/PERFORMANCE.md); the K-matrix correction is described in
[K-matrix force correction](docs/K_MATRIX.md).

## GPU installation

For a compatible CUDA 12/NVIDIA environment:

```sh
python -m pip install ".[gpu]"
pyib-run examples/cpu_fixed_sphere_smoke.json --device cuda --outdir runs/fixed-cuda
```

Equivalent dependency lists are `requirements-cpu.txt` and
`requirements-gpu.txt`. See [dependency details](docs/DEPENDENCIES.md).

## Pressure postprocessing

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

The command reconstructs the pressure field and corrects surface pressure.
Raw and corrected pressure values are saved separately. Outputs include
`pressure.npz`, `surface_pressure.csv`,
`postprocess.json` and optional ParaView VTU sample-point files.
The surface integral reports the pressure contribution to force. This
release contains no wall-shear recovery or surface-total-force output.
See [postprocessing notes](docs/POSTPROCESSING.md) for method details.

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess2 --two-sided --far-gauge 1.0
```

`--two-sided` evaluates the pressure on both sides of a triangulated surface,
open (zero-thickness plates, wings) or closed, with an optional far-field
pressure zero. The command without it is unchanged.

## Outputs and restart

Runs write configuration and provenance to `metadata.json`, time histories to
`history.csv`, restart state to `checkpoint.npz`, and the final field to
`final_field.npz` when enabled. Optional periodic fields go into `snapshots/`.

Resume using the same run directory and a larger target step count:

```sh
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed --resume runs/fixed/checkpoint.npz --steps 20
```

## Source checks

```sh
python tools/verify_source_manifest.py
python tools/run_tests.py
python tools/check_pressure_validation.py
```

The source manifest records both original and current hashes, including the
optional CPU kernel, its dispatch and execution-identity changes, and the
additions of 2026-11-17. With `solver.operator = "assembled"` and the K-matrix
correction off, the time stepping reproduces the earlier results bit for bit on
the bundled CPU examples. Packaging and test evidence is retained in `checks/`.

## Validation examples

The D32/D40 oscillating-sphere simulations completed six periods using
CUDA and float64. The cycle 5→6 force waveform difference is below 0.035%
of the analytical force amplitude at both resolutions. See the
[pressure and runtime-force results](docs/VALIDATION.md),
[machine-readable summary](checks/pressure-validation-20260925.json) and
[validation evidence](checks/pressure-validation-20260925.zip).
Both runs were repeated in full with the current source in November 2026 on
an 8 GB workstation GPU and gave the same error figures
([record](checks/validation-reruns-20261122.json)).
The bundled small examples provide installation and execution checks;
the D32/D40 results provide the numerical benchmark evidence.

## Retained manuscript

The [original manuscript draft](paper/IB_GPU_Solver_Paper_CN_JCP.pdf) is kept
unchanged from the existing repository, including its contents and original
author information. Release archives (the source distribution and the source
archives of a release) do not contain the PDF.
