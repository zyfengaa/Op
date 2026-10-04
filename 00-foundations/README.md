# 00 · 基础：先能解释，再写 Kernel

这一阶段不要求写 GPU kernel。目标是为后续代码建立共同语言，能从张量公式推导数据地址，能解释 CPU/GPU 的差异，并知道框架如何调用算子。

## 学习顺序

1. [CPU、GPU 与执行模型](computer-architecture.md)
2. [内存层级与数据移动](memory-hierarchy.md)
3. [算子在框架中的生命周期](framework-op-lifecycle.md)
4. [概念自测](quiz.md)

## 本阶段门槛

- 能解释 launch overhead 为什么让很小的 kernel 不一定比 CPU 快。
- 能从 row-major shape 和 stride 算出任意元素地址。
- 能区分通用并行概念和 CUDA 专属术语。
- 能说明一次 forward 算子调用可能涉及哪些检查和后端 dispatch。

建议投入 1–2 周，但以掌握检查点为准，不按日历强行升级。

## 参考资料

- [CUDA Programming Guide · Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CMU 15-418/618 Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s20/www/)
- [PyTorch ATen overview in source](https://github.com/pytorch/pytorch/tree/main/aten)
