# 实验入口、依赖与验证范围

所有命令从仓库根目录执行。正文中的代码片段用于定位思路；这里链接的是可运行入口。

## 标准 Python：数学与数据流

~~~powershell
python examples/cpu/dataflow_lab.py
python -m unittest discover -s examples/cpu -v
~~~

只需要 Python 标准库，按 [CPU 索引](cpu/README.md) 可运行单个算子。dataflow_lab 会输出线程负责的列、Welford 状态、GEMM 的共享 tile 与累加器、Attention 的 m/l/o 以及 im2col 矩阵。

本次修订在本机执行了 17 项 unittest，全部通过。它们检验数学、索引模拟、不同分块的结果一致性和部分无效输入，不检验 GPU 的内存竞争、同步或速度。

## PyTorch：真实框架注册和梯度

~~~powershell
python examples/pytorch/custom_bias_relu.py
~~~

源文件为 [custom_bias_relu.py](pytorch/custom_bias_relu.py)。本机 PyTorch 2.8.0+cpu 下，forward、gradcheck、opcheck、fullgraph capture、非连续输入和错误形状检查均通过。图捕获使用 eager backend，验证兼容性，不代表实现了编译加速或融合 CUDA kernel。

示例需要版本提供 torch.library.custom_op、register_fake、register_autograd、opcheck 等接口。若本机版本缺少 API，先核对官方教程与版本；不要把删掉检查当作集成完成。

## 可判定练习与模型切片

~~~powershell
python exercises/check.py selftest
python exercises/reading/check.py --answers exercises/reading/solutions.json
python -m unittest discover -s 06-model-slice -p "test_*.py" -v
~~~

前两项只需要标准库：故障自检必须证明 12 个错误确实被用例触发，并检查手算锚点；阅读题参考解应为 15/15。学生默认答案故意未修复/未填写，不能把学生的初始失败当作仓库回归失败。模型切片需要 PyTorch，7 项测试覆盖完整/分段/逐 token 对齐及缓存语义。

[算子实测脚本](../labs/operator_story.py) 和 [模型实测脚本](../06-model-slice/run.py) 分别保存自己的原始报告。读 [算子解读](../labs/results/cpu-reference/README.md)、[模型解读](../06-model-slice/results/cpu-reference/README.md) 后再比较自己的结果，不要求重现相同微秒数。

## CUDA：设备实现与计时

~~~powershell
cmake -S examples/cuda -B build/cuda
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
.\build\cuda\Release\row_ops.exe --benchmark
~~~

[构建入口](cuda/CMakeLists.txt) 包含六个 executable：Vector Add、单 block Reduce、Naive GEMM、Tiled GEMM、[row_ops](cuda/row_ops.cu) 和 [operator_ladders](cuda/operator_ladders.cu)。row_ops 包括 FP32 连续行 Softmax/LayerNorm、双精度 CPU reference、24 组算子/宽度检查，以及 CUDA event 计时。operator_ladders 实际实现 Add/Sum 各三个版本，检查 7 种长度（含 host 空输入分支），并完整计时两阶段归约。

Linux 单配置生成器通常在 build/cuda/ 下生成程序。默认以 native 架构配置，目标 GPU 不同时应显式传入正确的 CMAKE_CUDA_ARCHITECTURES。--profile 用于固定 128×1024 输入的 profiler 采集，详见 [性能测量章节](../02-perf-aware-ops/benchmark-and-profiling.md)。

本次修订环境没有 nvcc，PyTorch 也没有 CUDA 后端，因此这些 CUDA 源码尚未在本机编译/运行，sanitizer 和性能结果均待设备验证。提供源码、reference 和执行入口不等于 GPU 测试已经通过。

## 实现边界

- Attention 和 Conv2D 进阶实验是简化 CPU 算法；没有手写 GPU FlashAttention/完整 Conv2D 实现。模型切片调用 PyTorch SDPA 自动分派，本机 trace 是 CPU 路径。
- GEMM V3 提供对齐、向量分工和验证设计，尚无设备验证过的向量化 kernel。
- 国产后端的设备与 SDK 尚未确定，迁移章节提供方法和契约分析，未提供虚构的厂商 API。
- 性能演算数字在正文标为假设或示例；实际设备成绩应单独记录。

## 参考资料

- [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)：核对注册、fake、autograd 和测试接口。
- [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)：选择构建架构。
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：设备执行和工具链基础。
