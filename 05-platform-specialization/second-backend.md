# 做一次第二后端：Python reference → PyTorch 原生 CPU

这次实际运行同一个逻辑契约的两个后端，不需要先拥有另一块加速卡。Python 逐元素/顺序循环是 reference；PyTorch CPU add/sum 是原生实现后端。它不是手写 SIMD 教程，也没有验证最终用了哪条向量指令。

## 已做了什么

[labs/operator_story.py](../labs/operator_story.py) 使用有限 float64 输入，两个后端分别执行 Vector Add 与 Sum，并检查逐元素/标量结果一致。输入包含正负值和不同长度，空输入语义在故障/CPU 基础实验中另有练习。

同一脚本比较 resident tensor 和“从列表转入再转回”的两种路径，把适配层成本显式暴露出来。运行：

~~~powershell
python labs/operator_story.py --output labs/results/local
~~~

已有 [本机原始数据](../labs/results/cpu-reference/summary.json) 与 [解读](../labs/results/cpu-reference/README.md)，可以实际比较自己的环境，而非填写假设表。

## 逐项对照

| 层次 | Python reference | PyTorch CPU native | CUDA 教学版本 |
|---|---|---|---|
| 公共语义 | add/sum、有限值 | 同一组逻辑输入输出 | add/sum；本例 FP32 |
| 存储 | Python float 列表 | 连续 float64 tensor | 设备 float 数组 |
| 任务组织 | Python 循环 | 库内部调度 | 单线程 / 元素线程 / grid-stride / 两阶段 |
| 边界 | 列表长度 | tensor shape/stride | 显式 ceil、mask、partial |
| 同步 | 同步调用 | 本实验 CPU 同步 | stream、事件/同步点 |
| 验证状态 | 本机通过 | 本机通过 | 源码已提供，本机无 nvcc 待测 |

CUDA 的实现见 [operator_ladders.cu](../examples/cuda/operator_ladders.cu)，Triton 的 Vector Add 教学入口仍在第一阶段。这里没有把 dtype 不同的计时混成跨设备排名。

## 这次迁移的真正收获

函数名改成 torch.add 很容易，契约和调用成本才需要解释。若上游数据本来是 tensor，转换版本不是正确的热路径；若上游每次只给几个 Python 数，resident-only 基准又低估了业务成本。本机数据确实出现小规模 Python 更快、大规模原生库更快的反转。

把这个经验带到 GPU/NPU 时，要重新确认输入驻留位置、布局转换、调度与同步。公共 reference 和反例输入仍可保留，平台实现应独立检查。

## 下一步练习

为非连续 PyTorch 视图加入相同语义测试，再尝试显式 contiguous 后计算，分别记录复制和计算时间。不要直接假设转换一定有收益。移植到目标设备时，先让 VA/RD 的边界输入通过，再测试自己的新平台特有条件。

## 参考资料

- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：线程、输入准备与计时范围。
- [PyTorch add](https://docs.pytorch.org/docs/stable/generated/torch.add.html)：广播、dtype 和输出语义。
- [PyTorch sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html)：归约维度与 dtype。
