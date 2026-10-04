# 02 · 有性能意识的算子

本阶段学习如何从 profile 指标形成可证伪的性能假设。核心不是“哪个百分比低”，而是工作量、数据流和硬件映射是否匹配。

## 课程

- [Softmax](01-softmax/README.md)：稳定公式、分阶段 reduction、融合和 online 算法。
- [LayerNorm](02-layernorm/README.md)：均值/方差、数值误差、行归约和融合。
- [GEMM](03-gemm/README.md)：M/N/K、naive → tiling → vectorized load → benchmark。
- [性能测量实战](benchmark-and-profiling.md)：CUDA events、计时范围、Roofline、Nsight 与失败假设分析。

## 每个优化版本必须留下

1. 正确性结果和测试 shape。
2. 编译器、设备、dtype、输入布局。
3. baseline 与优化后 kernel 时间。
4. profiler 证据及指标含义。
5. “观察 → 假设 → 改动 → 结果 → 是否保留”的记录。

参考：[NVIDIA Nsight Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/) · [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
