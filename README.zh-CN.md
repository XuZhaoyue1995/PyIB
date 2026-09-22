# PyIB — A Python Immersed Boundary Solver for CPUs and GPUs

[English](README.md)

PyIB 是独立的 Python 浸没边界求解器，支持 NumPy/SciPy CPU 和 CuPy CUDA GPU
后端，提供 JSON 算例配置、自动网格生成、断点续算、结果检查以及压力和
表面应力后处理。CFDAgent 可以通过命令行接口调用 PyIB。

**软件唯一作者：Zhaoyue Xu。** 见 [作者信息](AUTHORS.md) 和
[软件引用信息](CITATION.cff)。Python 分发包和导入名为 `pyib`；
当前研究预发布版本为 `0.1.0rc1`。

软件许可证仍待定，本次预发布未授予开源许可。详见 [发布状态](RELEASE_STATUS.md)。

## CPU 安装与运行

在 Python 3.10+ 的独立环境中，进入本目录执行：

```sh
python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed
pyib-validate runs/fixed
```

随包 JSON 示例包含固定球、振荡球，以及固定球 Re=100 D8/D16 配置。

## GPU 安装与运行

使用相容的 CUDA 12/NVIDIA 环境：

```sh
python -m pip install ".[gpu]"
pyib-run examples/cpu_fixed_sphere_smoke.json --device cuda --outdir runs/fixed-cuda
```

CPU/GPU 依赖清单分别为 `requirements-cpu.txt` 和 `requirements-gpu.txt`。
详细说明见 [依赖文档](docs/DEPENDENCIES.md)。

## 压力与表面应力后处理

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

该命令完成压力场重构、表面压力修正和壁面切应力提取，分别保存原始值与修正值。
输出包括 `pressure_and_stress.npz`、`surface_stress.csv`、`postprocess.json`，
并可导出 ParaView 可读的 VTU 采样点文件。方法细节见 [后处理说明](docs/POSTPROCESSING.md)。

## 结果与断点续算

运行目录中的 `metadata.json` 保存配置和来源，`history.csv` 保存时间历史，
`checkpoint.npz` 保存续算状态，启用场输出时生成 `final_field.npz`。
可选的周期场快照位于 `snapshots/`。

在同一运行目录中增加目标步数即可续算：

```sh
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed --resume runs/fixed/checkpoint.npz --steps 20
```

## 源码检查

```sh
python tools/verify_source_manifest.py
python tools/run_tests.py
```

源码清单用于识别数值程序版本；详细安装、集成及测试记录保存在 `checks/`。

## 保留的论文草稿

仓库保留 [原版论文工作稿](paper/IB_GPU_Solver_Paper_CN_JCP.pdf)，
其内容和原有作者信息均不作修改。
