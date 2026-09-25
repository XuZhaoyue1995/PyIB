# PyIB — A Python Immersed Boundary Solver for CPUs and GPUs

[English](README.md)

PyIB 是独立的 Python 浸没边界求解器，支持 NumPy/SciPy CPU 和 CuPy CUDA GPU
后端，提供 JSON 算例配置、自动网格生成、断点续算、结果检查、压力场重构及
表面压力修正。CFDAgent 可以通过命令行接口调用 PyIB。

**求解器与压力模块已完成 D32/D40 振荡球基准算例（Re₀ = 0.2）的严格数值核验。**
D40 的运行受力谐波向量误差为 **4.1190%**，修正表面压力的面积、时间综合相对
L₂ 误差为 **2.2224%**。核验包括解析解对照、周期收敛、网格与时间步联合加密，
以及源码和数据来源核对。详见 [完整验证记录](docs/VALIDATION.md)。

**软件唯一作者：Zhaoyue Xu。** 见 [作者信息](AUTHORS.md) 和
[软件引用信息](CITATION.cff)。Python 分发包和导入名为 `pyib`；
当前研究预发布版本为 `0.1.0rc2`。

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

## 压力后处理

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

该命令完成压力场重构与表面压力修正，分别保存原始压力和修正压力。
输出包括 `pressure.npz`、`surface_pressure.csv`、`postprocess.json`，
并可导出 ParaView 可读的 VTU 采样点文件。表面积分给出压力分力；本版不包含
壁面切应力重构及表面积分总力输出。方法细节见 [后处理说明](docs/POSTPROCESSING.md)。

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
python tools/check_pressure_validation.py
```

源码清单记录 13 个保持原样的数值文件和仅处理压力的后处理程序。
CPU/CUDA 时间推进求解器及压力数值方法保持不变；安装与测试记录保存在 `checks/`。

## 验证算例

D32/D40 振荡球使用 CUDA、float64 完成六周期计算。两个分辨率下，第 5→6 周期
受力波形差异均小于解析受力幅值的 0.035%。详见
[压力与运行受力验证](docs/VALIDATION.md)、
[机器可读结果](checks/pressure-validation-20260925.json) 和
[验证数据包](checks/pressure-validation-20260925.zip)。随包的小型示例用于检查安装和
运行，D32/D40 结果提供数值基准验证依据。

## 保留的论文草稿

仓库保留 [原版论文工作稿](paper/IB_GPU_Solver_Paper_CN_JCP.pdf)，
其内容和原有作者信息均不作修改。
