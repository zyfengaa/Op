# Op 算子开发知识库

面向有普通开发经验、但没有系统做过 GPU/NPU 算子开发的人。目标不是“读过”，而是能解释正确性、性能和硬件行为，并能审查 AI 生成的实现。

这个仓库同时服务两种用法：**学**的人顺着流程主线走，**查**的人直接进知识域。两条路径共用同一份内容，不重复维护。

## 三条使用路径

| 你要做什么 | 从哪进 | 说明 |
|---|---|---|
| **学**：系统建立算子开发能力 | [流程主线](01-workflow/README.md) | 一个算子从需求到交付的九个阶段，每步给证据和判据 |
| **查**：手头有个具体问题 | [目录结构](#目录结构) 或 [术语表](99-reference/glossary.md) | 按知识域直达，不用按课程顺序 |
| **练**：动手验证理解 | [练习层](09-practice/README.md) | 12 个故障题、15 道阅读题、算子五件套、完整模型切片 |

第一次来，建议先读 [流程主线](01-workflow/README.md)——它给出全局地图，之后无论学还是查都不容易迷路。环境从 [00-start](00-start/README.md) 开始。

## 目录结构

~~~text
00-start/           入口：前置知识、环境搭建、学习路径
01-workflow/        流程主线：一个算子从需求到交付的九个阶段
02-foundations/     基础域：执行模型、内存层级、索引与布局、精度契约
03-operators/       算子域：按算子类型组织，每类自带知识、推导、实现、反例
04-performance/     性能域：测量、分析、工具链、优化手法、性能反例
05-frameworks/      框架域：算子生命周期、后端接入、测试
06-platforms/       平台域：哪些能力可迁移、迁移时哪些必须重学
07-quantization/    量化域：低精度、量化算法、量化 kernel、精度评估
08-engineering/     交付域：CI、端到端项目
09-practice/        练习层：故障注入、代码阅读、算子故事、模型切片
99-reference/       速查层：术语表、清单、模板、外部资料
assets/             可运行代码与构建入口
~~~

**主线是时间轴**，告诉你下一步做什么；**知识域是空间索引**，告诉你具体知识在哪。每个阶段页链接到对应知识域，每个知识域页反向标注它服务于哪个阶段。

## 算子域怎么读

每个算子目录内部结构统一：

~~~text
03-operators/<类别>/<算子>/
  README.md        知识：定义、契约、陷阱、性能要点   ← 查阅入口
  walkthrough.md   推导：手算、数据流、逐段代码       ← 学习入口
  impl/            实现：该算子特有的 CUDA 源码
  case-study.md    五件套：从需求到交付的完整记录     （部分算子有）
~~~

跨算子共享的代码（CPU 教学实验、服务多个算子的 `.cu`、CMake 入口）统一放在 [assets/](assets/README.md)，按语言组织。所以 attention、convolution 等目录没有 `impl/`，它们的实验代码在 `assets/cpu/`。

同一个算子，查的人读 README，学的人读 walkthrough，踩坑的人对照 impl 和反例。

## 无 GPU 也能开始的实验

仓库的 CUDA 代码需要 NVIDIA GPU 和 CUDA Toolkit；其余部分只需要标准 Python，装有 PyTorch 后范围更大。[00-start](00-start/README.md) 里有完整的能力矩阵：哪些章节能做完、哪些只能读、哪些必须等设备。

~~~powershell
# 故障注入练习（只需标准 Python）
python 09-practice/check.py list
python 09-practice/check.py show VA01
python 09-practice/check.py grade VA01

# CPU 教学实验
python assets/cpu/indexing_lab.py
python assets/cpu/softmax_lab.py
python -m unittest discover -s assets/cpu

# 需要 PyTorch
python 09-practice/operator-stories/operator_story.py --output 09-practice/operator-stories/results/local
python 09-practice/model-slice/run.py --profile --output 09-practice/model-slice/results/local
~~~

## CUDA 示例

需要兼容的 NVIDIA GPU、CUDA Toolkit 和 CMake：

~~~powershell
cmake -S assets/cuda -B build/cuda
cmake --build build/cuda --config Release
.\build\cuda\Release\vector_add.exe
.\build\cuda\Release\row_ops.exe --benchmark
ctest --test-dir build/cuda -C Release --output-on-failure
~~~

## AI 辅助的学习方法

AI 可加快样板代码、改写和测试草稿；学习者仍需检查索引、mask、同步、dtype、数值边界和设备假设，并亲自测量性能。使用 [AI 审查清单](99-reference/checklists/ai-usage-policy.md)，在 [实验记录模板](99-reference/templates/experiment-log-template.md) 中留下假设和证据。不要把 AI 写出的性能数字当成实测。

## 证据和限制

- 课程中的模拟数据都应明确标注为模拟，不代表具体硬件成绩。
- CPU 实测报告可复跑；模型切片通过 7 项测试。模型使用随机权重，不是完整预训练语言模型，也没有手写 GPU FlashAttention。
- CUDA 示例源代码和 CMake 入口已提供，但**本仓当前环境没有 nvcc 和 GPU，`.cu` 文件未经编译验证**。实际编译需在有 CUDA Toolkit 的设备上完成。
- Triton 教学代码按官方接口编写，需在本机兼容版本和 GPU 上验证。
- 国产加速卡的 API、工具和约束随 SDK/设备变化；以厂商对应版本的官方资料为准。

## 外部资料

按主题查看 [外部资料索引](99-reference/external-links.md)：CUDA、Triton、Nsight、PyTorch、CUTLASS 和进阶论文。
