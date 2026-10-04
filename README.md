# Op 算子开发学习仓

这是一套面向有普通开发经验、但没有系统做过 GPU/NPU 算子开发的课程。学习目标是能解释正确性、性能和硬件行为，并能审查 AI 生成的实现。

课程采用阶段门槛：先学习通用概念，再以 CUDA 为第一个可运行后端，后续学习 Triton、Attention 和平台迁移。阶段周期仅作参考，以验收清单为准。

## 课程主线

| 阶段 | 目录 | 学习内容 | 阶段产出 |
|---|---|---|---|
| 00 基础 | [00-foundations](00-foundations/README.md) | CPU/GPU、SIMT、内存层级、框架算子生命周期 | 能推导地址并解释 launch 和带宽 |
| 01 Hello World | [01-hello-world-ops](01-hello-world-ops/README.md) | Vector Add、Activation、Reduction | reference、kernel、边界测试、baseline |
| 02 性能意识 | [02-perf-aware-ops](02-perf-aware-ops/README.md) | Softmax、LayerNorm、GEMM、Roofline | profiling 证据链和优化实验 |
| 03 进阶算子 | [03-advanced-ops](03-advanced-ops/README.md) | FlashAttention、Conv2D、Fusion | 组合算法、数据流和硬件映射 |
| 04 工程化 | [04-engineering](04-engineering/README.md) | 接口、测试、CI、复现 | 可维护、可集成、可回归的算子 |
| 05 平台专精 | [05-platform-specialization](05-platform-specialization/README.md) | CUDA/Triton/国产后端迁移 | 保留公共语义，重做平台实现 |

## 新人从这里开始

1. 阅读 [基础执行模型](00-foundations/computer-architecture.md) 和 [内存层级](00-foundations/memory-hierarchy.md)。
2. 完成 [Vector Add 教程](01-hello-world-ops/01-vector-add/README.md)，编译运行 CUDA 示例。
3. 学习 [Elementwise Activation](01-hello-world-ops/02-elementwise-activation/README.md) 和 [Reduction](01-hello-world-ops/03-reduction-sum-max/README.md)。
4. 进入 [Softmax、LayerNorm 和 GEMM](02-perf-aware-ops/README.md)。
5. 用 [阶段验收清单](checklist/stage-gates.md) 判断是否进入下一阶段。

## 目录说明

~~~text
00-foundations/                  概念地基和自测
01-hello-world-ops/              第一个平台上的基础算子
02-perf-aware-ops/               性能感知案例和 GEMM 版本阶梯
03-advanced-ops/                 Attention、卷积、融合
04-engineering/                  接口、测试和 CI
05-platform-specialization/      后端专精和迁移
checklist/                       阶段门槛、AI 和实验记录模板
references/                      官方文档、论文和 profiler 模板
examples/                        可运行脚本及构建入口
archive-legacy-docs/              旧版摘要材料，保留供查阅，不作为课程入口
~~~

## 案例运行

标准 Python 示例不需要第三方库：

~~~powershell
python examples/gemm_learning_demo.py
python examples/simulate_gemm_analysis.py
~~~

CUDA 示例需要兼容的 NVIDIA GPU、CUDA Toolkit 和 CMake：

~~~powershell
cmake -S examples/cuda -B build/cuda
cmake --build build/cuda --config Release
~~~

CUDA 程序：

~~~powershell
.\build\cuda\Release\vector_add.exe
.\build\cuda\Release\block_reduce_sum.exe
.\build\cuda\Release\naive_gemm.exe
.\build\cuda\Release\tiled_gemm.exe
~~~

Linux/macOS 通常在 build/cuda/ 下运行同名程序。Triton 示例位于 Vector Add 课程目录，需要按 Triton 当前安装指南准备 Python/GPU 环境。

## AI 辅助的学习方法

AI 可加快样板代码、改写和测试草稿；学习者仍需检查索引、mask、同步、dtype、数值边界和设备假设，并亲自测量性能。使用 [AI 审查清单](checklist/ai-usage-policy.md)，在 [实验记录](checklist/experiment-log-template.md) 中留下假设和证据。不要把 AI 写出的性能数字当成实测。

## 证据和限制

- 课程中的模拟数据都应明确标注为模拟，不代表具体硬件成绩。
- CUDA 示例源代码和 CMake 入口已提供；实际编译需在有 CUDA Toolkit 的设备上完成。
- Triton 教学代码按官方接口编写，需在本机兼容版本和 GPU 上验证。
- 国产加速卡的 API、工具和约束随 SDK/设备变化；以厂商对应版本的官方资料为准。

## 参考资料入口

按主题查看 [references 索引](references/README.md)：CUDA、Triton、Nsight、PyTorch、CUTLASS 和进阶论文。每个章节也附有适用于该章的资料。
