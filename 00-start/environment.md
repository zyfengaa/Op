# 环境搭建与能力矩阵

## 三档环境

| 档位 | 需要什么 | 能做什么 |
|---|---|---|
| **A · 标准 Python** | Python 3.10+，无第三方依赖 | 故障注入练习、代码阅读题、CPU 教学实验（`assets/cpu/`） |
| **B · 加 PyTorch** | A + 一个能 `import torch` 的环境（CPU 版即可） | 算子故事实测、模型切片、框架接入实验 |
| **C · 加 CUDA** | B + NVIDIA GPU + CUDA Toolkit + CMake | 编译运行 `assets/cuda/` 的所有 CUDA 示例 |

只装 A 也能走完大部分知识域。C 只影响**能不能得到真实硬件数字**，不影响能不能理解内容。

## 无 GPU 时的能力矩阵

这是最需要提前知道的一页。

| 内容 | 无 GPU 时 | 说明 |
|---|---|---|
| [流程主线](../01-workflow/README.md) | **可完成** | 流程和方法不依赖设备 |
| [基础域](../02-foundations/README.md) | **可完成** | 概念与手算 |
| [故障注入练习](../09-practice/fault_injection) | **可完成** | 纯 Python 数据流模拟，12 题 39 组输入 |
| [代码阅读题](../09-practice/reading) | **可完成** | 15 道判定题 |
| [CPU 教学实验](../assets/cpu/README.md) | **可完成** | 算法正确，但不模拟 GPU 速度 |
| [算子故事实测](../09-practice/operator-stories/README.md) | **可完成**（需 PyTorch） | 得到的是 CPU 数字 |
| [模型切片](../09-practice/model-slice/README.md) | **可完成**（需 PyTorch） | SDPA 走 CPU 路径，不是 FlashAttention |
| 各类算子的 `impl/*.cu` | **只能读** | 源码可读、逻辑可推，但未编译验证 |
| CUDA 性能章节的实测部分 | **只能读** | 需要真实设备的带宽、occupancy、stall 数据 |
| [性能手法](../04-performance/techniques/) 的效果验证 | **只能读** | 手法的适用条件可理解，收益必须自己在设备上测 |

**关键纪律**：无 GPU 时得到的 CPU 数字，**不能**用来推断 HBM 带宽、bank conflict、寄存器数或 Tensor Core 利用率。这些量只在设备上才有意义。看到 CPU 报告时，先确认它测的是什么。

## A 档：验证安装

~~~powershell
python --version                 # 3.10 以上
python 09-practice/check.py list # 能列出 VA01..AT01 就正常
python -m unittest discover -s assets/cpu
~~~

## B 档：加 PyTorch

任意能 `import torch` 的环境均可，CPU 版足够。验证：

~~~powershell
python -c "import torch; print(torch.__version__)"
python 09-practice/operator-stories/operator_story.py --output 09-practice/operator-stories/results/local
~~~

脚本会先检查结果，再采样计时，最后写入 `summary.json`、profiler 表和 trace。报告可能包含本地路径，分享前先看内容。

## C 档：加 CUDA

需要三样：NVIDIA GPU、CUDA Toolkit（`nvcc`）、CMake。

~~~powershell
nvcc --version
cmake -S assets/cuda -B build/cuda
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
~~~

`assets/cuda/CMakeLists.txt` 里的 `CMAKE_CUDA_ARCHITECTURES` 是**可覆盖**的：未指定时取 `native`，需要交叉编译或指定架构时用 `-DCMAKE_CUDA_ARCHITECTURES=90` 之类显式传入。

进阶工具（[性能工具链](../04-performance/tooling.md)）：

| 工具 | 用途 |
|---|---|
| `ncu`（Nsight Compute） | 单个 kernel 的吞吐、occupancy、stall 原因 |
| `nsys`（Nsight Systems） | 时间线、kernel 间空隙、host↔device 行为 |
| `compute-sanitizer` | 越界、竞争、同步错误 |
| `nvcc / cuobjdump / nvdisasm` | 生成和阅读 SASS |

## 已知的环境陷阱

- **找不到 torch**：使用已经装好兼容 PyTorch 的环境，不必重建。
- **初始练习失败**：`student.py` 是待修复版本，判题失败是设计，不是环境问题。
- **微秒数每次不同**：先核对采样条件和系统负载。不要同时跑多个 benchmark，也不要一边开 profiler 一边把耗时当普通运行结果。
- **NumPy 未安装的提示**：本仓的实验脚本不调用 NumPy，可以忽略；这不等于任何依赖警告都可以忽略。
