# 无 GPU 环境也能运行的概念实验

这些脚本只依赖 Python 标准库（建议 Python 3.10+）。它们演示数学语义、线程索引、padding、数值稳定和 tile 数据复用；CPU 运行时间不能用于预测 CUDA kernel 性能。

从仓库根目录执行：

~~~powershell
python examples/cpu/indexing_lab.py
python examples/cpu/elementwise_lab.py
python examples/cpu/reduction_lab.py
python examples/cpu/softmax_lab.py
python examples/cpu/layernorm_lab.py
python examples/cpu/gemm_lab.py
python examples/cpu/advanced_lab.py
python -m unittest discover -s examples/cpu
~~~

## 对应课程

| 脚本 | 对应课程 | 主要观察 |
|---|---|---|
| indexing_lab.py | 00 基础 | block/grid、无效线程、stride |
| elementwise_lab.py | 01 Vector Add/Activation | 独立输出、数值极值 |
| reduction_lab.py | 01 Reduction | padding identity、归约轮次 |
| softmax_lab.py | 02 Softmax | 减 max、在线归一化状态 |
| layernorm_lab.py | 02 LayerNorm | 均值、方差、常量行 |
| gemm_lab.py | 02 GEMM | 非整除 tile、A/B 复用 |
| advanced_lab.py | 03 进阶算子 | Attention、Conv2D、bias+ReLU 的简化 reference |

每个脚本先读公式，再手算小输入，最后运行并核对输出。请把观察写进实验日志，不要只看 PASS。进阶案例只是简化语义 reference，不是 GPU 高性能实现。
