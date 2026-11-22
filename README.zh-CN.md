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
当前发布版本为 `0.2.0`。

PyIB 以 [BSD 3-Clause 许可证](LICENSE) 发布。`paper/` 中保留的论文稿不在该许可证
范围内（[说明](paper/README.md)）。在发表的工作中使用 PyIB 时，请按
[CITATION.cff](CITATION.cff) 引用。另见 [发布状态](RELEASE_STATUS.md)。

**`0.2.0`（2026-11-22）相对 `0.1.0rc2` 的新增内容：** Helmholtz 算子提速
（D16 基准算例实测 CUDA 2.2 倍、CPU 3.6 倍）；从 IBFast 移植了可选的 K 矩阵
修正力（默认关闭）和三角剖分面的两侧表面压力修正。D32/D40 六周期验证算例已用
改动后的代码完整重跑，误差与存档一致。详见 [变更记录](CHANGELOG.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/pyib/` | 软件包：网格生成器和 `solver/`（时间推进、浸没边界、K 矩阵修正力、压力后处理、结果校验器） |
| `examples/` | 用于检查安装和运行的小型 JSON 算例 |
| `tests/` | 单元测试（`python tools/run_tests.py`） |
| `tools/` | 源码清单校验、测试入口、验证证据校验 |
| `docs/` | 方法与实测说明，见下面的文档列表 |
| `checks/` | 按日期保存的运行与实测记录（[索引](checks/README.md)） |
| `paper/` | 保留的原版论文工作稿 |

文档：[验证记录](docs/VALIDATION.md) · [压力后处理](docs/POSTPROCESSING.md) ·
[K 矩阵修正力](docs/K_MATRIX.md) · [性能说明](docs/PERFORMANCE.md) ·
[CPU 加速](docs/CPU_ACCELERATION.md) · [依赖](docs/DEPENDENCIES.md) ·
[变更记录](CHANGELOG.md) · [发布状态](RELEASE_STATUS.md)

## CPU 安装与运行

在 Python 3.10+ 的独立环境中，进入本目录执行：

```sh
python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed
pyib-validate runs/fixed
```

随包 JSON 示例包含固定球、振荡球、固定球 Re=100 D8/D16 配置，以及一个打开了
K 矩阵修正力的小算例（`cpu_oscillating_sphere_d8_k_matrix.json`）。

### 可选的 CPU 并行稀疏乘法

在 Linux 上使用兼容 GCC 的 OpenMP 编译器，于安装时构建可选扩展：

```sh
PYIB_BUILD_OPENMP=1 python -m pip install .
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed-openmp --cpu-sparse-backend openmp --cpu-sparse-threads 4
```

普通安装仍使用 SciPy。扩展将 CPU float64 的 CSR 矩阵乘法按行并行，
不改变流动模型、CG 算法及 GPU 内核。配置、验证范围与续算兼容性见
[CPU 加速说明](docs/CPU_ACCELERATION.md)。
源码回归已通过 23 项单元测试，以及固定球、运动球的多步与续算对照，
被比较的数值状态完全一致。

### 不需要编译器的多线程

```sh
pyib-run examples/cpu_fixed_sphere_smoke.json --outdir runs/fixed-threads --cpu-sparse-backend threads --cpu-sparse-threads 8
```

`threads` 后端把 SciPy 自己的 CSR 内核按行块放到线程池上执行，不需要扩展，
任何平台可用，结果与 SciPy 逐位相同。

## 0.2.0 新增的求解器选项

| 选项 | 默认 | 作用 |
|---|---|---|
| `solver.operator`（`--operator`） | `"factored"` | Helmholtz 算子写成三个稀疏因子（每条边约 23 个非零，原来 120 个）：同一个算子，只差舍入，内存约五分之一。`"assembled"` 在 CPU 上与以前的结果逐位相同。 |
| `solver.cg_reference_speed`（`--cg-reference-speed`） | `0`（关） | 物体在静止流体中运动时，给 CG 停机判据加一个与参考系无关的下限；迭代数减少（基准算例整体快 1.7–2.9 倍）；六周期内基准算例受力的变化为其最大值的 6e-7（D16）至 7e-6（D40）。 |
| `ib.k_matrix.enabled`（`--k-matrix on`） | `false` | 每个内迭代把动量方程和无滑移条件联立求解（IBFast `ib_update='pcg'`）。 |
| `output.residual_state` | `"auto"` | 流场快照保存哪个动量残差：末次内迭代的（显式更新）或时间步结束时的（K 矩阵）。 |

实测耗时及与旧结果的对比见 [性能说明](docs/PERFORMANCE.md)；K 矩阵修正力见
[K 矩阵修正力说明](docs/K_MATRIX.md)。

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

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess2 --two-sided --far-gauge 1.0
```

`--two-sided` 给出三角剖分面两侧的压力，开口面（零厚度板、翅）和闭合面都适用，
可选远场压力零点。不加该参数时命令的行为不变。

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

源码清单记录原始与当前哈希，包括可选 CPU 内核、调用接口和执行身份改动，以及
2026-11-17 的新增内容。`solver.operator = "assembled"` 且 K 矩阵修正关闭时，时间
推进在随包 CPU 算例上与以前的结果逐位相同；安装与测试记录保存在 `checks/`。

## 验证算例

D32/D40 振荡球使用 CUDA、float64 完成六周期计算。两个分辨率下，第 5→6 周期
受力波形差异均小于解析受力幅值的 0.035%。详见
[压力与运行受力验证](docs/VALIDATION.md)、
[机器可读结果](checks/pressure-validation-20260925.json) 和
[验证数据包](checks/pressure-validation-20260925.zip)。2026 年 11 月用当前源码在一块
8 GB 工作站显卡上完整重跑了这两个算例，误差数字相同
（[记录](checks/validation-reruns-20261122.json)）。随包的小型示例用于检查安装和
运行，D32/D40 结果提供数值基准验证依据。

## 保留的论文草稿

仓库保留 [原版论文工作稿](paper/IB_GPU_Solver_Paper_CN_JCP.pdf)，
其内容和原有作者信息均不作修改。发布包（源码分发包和各版本的源码压缩包）不含该 PDF。
