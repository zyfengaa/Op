# 07：GEMM 优化、Tensor Core 与 Pipeline

## 本章目标

把 GEMM 优化从“不断试参数”变成逐层验证：coalescing → tiling → register tiling → MMA → double buffer。

## 分析顺序

先确定 M/N/K、dtype、layout 和 Arithmetic Intensity，再检查 tile、warp mapping、register、shared memory、occupancy、cache、访存和 pipeline。前面的信息决定理论应该怎么跑，后面的指标解释为什么实际上没有达到。

## Tensor Core 检查清单

- 输入 dtype 是否符合目标 MMA 指令？
- M/N/K 是否满足指令 tile 和对齐要求？
- A/B fragment layout 是否正确？
- 编译产物是否真的包含 MMA 指令？
- warp 是否持续有 MMA 工作，还是一直在等待 load？

Tensor Core utilization 低不一定意味着应直接优化 Tensor Core；如果 AI 低、访存不合并或 Long Scoreboard Stall 高，应该先解决数据供给。

## Occupancy 决策

记录 registers/thread、shared memory/block、threads/block 和 resident blocks。Occupancy 不是越高越好：减少寄存器可能降低数据复用，增加并行度也可能增加访存压力。最终以 kernel 时间和 stall reason 为准。

## Pipeline

double buffer 用两组 shared-memory buffer，让当前 tile 计算时预取下一 tile。每一次改动都要检查同步正确性、数据依赖和边界。若带宽已经接近上限，pipeline 只能隐藏延迟，不能创造额外带宽。

## 小作业

对同一个 GEMM 逐次记录：

```text
版本 | 时间 | GFLOPS | HBM | Tensor Core | Occupancy | Stall | 假设是否成立
```

要求每次只改一个主要变量，并保留能复现的编译参数。

## 参考资料

- [CUDA C++ Best Practices：Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Programming Guide：Tensor Cores](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
