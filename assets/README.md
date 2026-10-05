# assets：可运行代码与构建入口

本目录存放**跨算子共享**的可运行代码。算子特有的推导和说明在 [03-operators](../03-operators/README.md) 里，这里只放能跑的东西。

## 目录

~~~text
assets/
  cpu/    纯 Python 教学实验（线程索引模拟、逐元素、归约、softmax、layernorm、GEMM、数据流）
  cuda/   CUDA 示例与 CMake 构建入口
  README.md
~~~

## 为什么代码集中在这里

三个原因：

1. **测试脚本跨多个实验。** `cpu/test_operator_lab.py` 用一个测试跑多份实验，拆开会让导入路径碎片化。
2. **部分文件服务多个算子。** `cuda/row_ops.cu` 同时覆盖 softmax 和 layernorm；`cuda/operator_ladders.cu` 同时覆盖 add 和 sum。
3. **构建入口只能有一个。** CMake 需要一处统一的目标列表。

因此这里按**语言**组织，而不是按算子；对应关系由各算子页的链接给出。

## CPU 实验

只需要标准 Python：

~~~powershell
python assets/cpu/indexing_lab.py
python assets/cpu/elementwise_lab.py
python assets/cpu/reduction_lab.py
python assets/cpu/softmax_lab.py
python assets/cpu/layernorm_lab.py
python assets/cpu/gemm_lab.py
python assets/cpu/dataflow_lab.py
python -m unittest discover -s assets/cpu
~~~

这些程序验证**算法和数据流**，不模拟 GPU 速度。用它们可以手算对照、做边界测试，但不能从中推断任何硬件性能。

## CUDA 示例

需要 NVIDIA GPU、CUDA Toolkit 和 CMake：

~~~powershell
cmake -S assets/cuda -B build/cuda
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
~~~

程序：

~~~powershell
.\build\cuda\Release\vector_add.exe
.\build\cuda\Release\block_reduce_sum.exe
.\build\cuda\Release\naive_gemm.exe
.\build\cuda\Release\tiled_gemm.exe
.\build\cuda\Release\row_ops.exe --benchmark
.\build\cuda\Release\operator_ladders.exe --benchmark
~~~

`row_ops.exe --profile` 是**精简负载模式**（固定 128×1024），用于配合外部 profiler 采集；它测出的时间不能当作常规性能数据。

`CMAKE_CUDA_ARCHITECTURES` 可覆盖，未指定时取 `native`。

## 验证状态

| 部分 | 状态 |
|---|---|
| `cpu/*.py` | **已验证**，本机通过 |
| `cuda/*.cu` | **源码已提供，但本仓环境无 nvcc 和 GPU，未经编译验证** |

这条限制很重要：`cuda/` 下的代码是按正确的 CUDA 语义写的，但**没有在真实设备上编译运行过**。请在有 CUDA Toolkit 的机器上首编，并按 [`00-start/environment.md`](../00-start/environment.md) 的 C 档流程验证。

## 算子与文件的对应

| 算子页 | 相关文件 |
|---|---|
| [vector-add](../03-operators/elementwise/vector-add/README.md) | `cpu/elementwise_lab.py`、`cuda/operator_ladders.cu` |
| [activation](../03-operators/elementwise/activation/README.md) | `cpu/elementwise_lab.py` |
| [reduction](../03-operators/reduction/README.md) | `cpu/reduction_lab.py`、`cuda/operator_ladders.cu`、`cuda/block_reduce_sum.cu` |
| [softmax](../03-operators/normalization/softmax/README.md) | `cpu/softmax_lab.py`、`cpu/dataflow_lab.py`、`cuda/row_ops.cu` |
| [layernorm](../03-operators/normalization/layernorm/README.md) | `cpu/layernorm_lab.py`、`cuda/row_ops.cu` |
| [matmul](../03-operators/matmul/README.md) | `cpu/gemm_lab.py`、`cuda/` 下的三个 GEMM 版本 |
| [attention](../03-operators/attention/README.md) | `cpu/advanced_lab.py` |
| [data-movement](../03-operators/data-movement/README.md) | `cpu/indexing_lab.py` |

## 相关页面

- [00-start/environment.md](../00-start/environment.md)：三档环境与能力矩阵
- [流程主线](../01-workflow/README.md)
- [练习层](../09-practice/README.md)
