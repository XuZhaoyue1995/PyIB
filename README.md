# PyIB — A Python Immersed Boundary Solver for CPUs and GPUs

[简体中文](README.zh-CN.md)

PyIB is an independent Python immersed-boundary solver with NumPy/SciPy CPU
and CuPy CUDA backends. It provides JSON case configuration, automatic mesh
construction, checkpoint/restart, result checks, and offline pressure and
surface-stress postprocessing. CFDAgent can call PyIB through its command-line
interface.

**Sole software author: Zhaoyue Xu.** Software attribution is recorded in
[AUTHORS.md](AUTHORS.md) and [CITATION.cff](CITATION.cff). The Python distribution
and import namespace are `pyib`; this research prerelease is `0.1.0rc1`.

The software license is pending; this prerelease does not grant an open-source
license. See [release status](RELEASE_STATUS.md).

## CPU installation and first run

Use Python 3.10 or newer in a virtual environment, from this directory:

```sh
python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed
pyib-validate runs/fixed
```

The bundled JSON examples include fixed and oscillating spheres, plus
fixed-sphere Re=100 D8/D16 configurations.

## GPU installation

For a compatible CUDA 12/NVIDIA environment:

```sh
python -m pip install ".[gpu]"
pyib-run examples/cpu_fixed_sphere_smoke.json --device cuda --outdir runs/fixed-cuda
```

Equivalent dependency lists are `requirements-cpu.txt` and
`requirements-gpu.txt`. See [dependency details](docs/DEPENDENCIES.md).

## Pressure and surface-stress postprocessing

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

The command reconstructs the pressure field, corrects surface pressure, and
recovers wall shear stress. Raw and corrected values are saved separately.
Outputs include `pressure_and_stress.npz`, `surface_stress.csv`,
`postprocess.json` and optional ParaView VTU sample-point files.
See [postprocessing notes](docs/POSTPROCESSING.md) for method details.

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
```

The source manifest identifies the numerical snapshot. Detailed packaging,
integration and test evidence is retained in `checks/`.

## Retained manuscript

The [original manuscript draft](paper/IB_GPU_Solver_Paper_CN_JCP.pdf) is kept
unchanged from the existing repository, including its contents and original
author information.
