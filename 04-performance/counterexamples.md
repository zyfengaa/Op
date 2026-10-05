# 性能反例档案：一次正确优化为什么不总是更快

这里保存实测与适用条件，不用虚构数据填满 shape/device 表。当前只有这台 CPU 的记录；CUDA/NPU 的反例应在有真实设备后单独提交。

## CE-001：原生后端在小输入上输给 Python

[原始报告](../09-practice/operator-stories/results/cpu-reference/summary.json) 对同一逻辑有限 float64 输入比较 Python loop 与 PyTorch native。N=1 时 Add 分别为 0.550/3.070 us，原生路径更慢；N=65536 时为 7454.500/74.390 us，原生路径明显更快。

解释：调用和框架成本在极少工作时不可忽略；大输入时 Python 逐元素解释的成本突出。报告还单测转换路径，避免把“tensor 常驻”的胜利推广到“每次从列表来回转换”的业务。

这比较的是后端候选，不是同一个 CUDA kernel 的硬件报告。它训练的判断是：先固定契约、输入位置和计时范围，再讨论选择阈值。当前采样点不能准确定位 crossover，若业务依赖阈值，应在附近加密测量。

## CE-002：树的层数少，不代表 Python 执行更快

同一报告包含 sum/python_tree。树形算法把依赖深度降为对数级，却在单进程 Python 中创建多轮列表，完全没有启动并行线程。比较原始样本中的 python_loop、python_tree、torch_resident，说明算法并行性要由执行后端实现才能变成速度。

请勿用这组 CPU 结果否定 GPU block reduction；两者执行模型不同。它反驳的是“写成树状代码就自动并行”的推断。

## CE-003：完整 block 的候选优化尚无稳定提速证据

[模型报告](../09-practice/model-slice/results/cpu-reference/summary.json) 中 prefill reference/optimized 的中位数为 1624.690/1608.290 us，decode 为 834.490/814.180 us；但采样区间明显重叠。不能据微小中位数差异宣称稳定的 1%～3% 收益。

候选同时合并投影和切换 SDPA，需要消融才能解释哪部分有效。当前 tiny block、CPU、小工作集也不代表大模型或 GPU 结果。

## 新记录必须附带的材料

写明比较对象、shape/dtype/device、线程与版本、计时范围、原始样本、正确性结果、环境/源码标识、假设和不支持的推论。没有真实数字时保留为实验任务，不混入本档案的实测表。

## 参考资料

- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：可靠比较方法。
- [本仓性能测量](measurement.md)：证据、假设和整体收益。
