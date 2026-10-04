# 03 · 进阶算子

进入本阶段前，应能解释 reduction、GEMM tiling、数据布局和 profiler 基础读数。Attention、卷积和融合都建立在这些基础上。

## 课程顺序

1. [FlashAttention 思路](01-flash-attention/README.md)：online softmax、tile 和减少 HBM 中间结果。
2. [Conv2D 实现路线](02-conv2d/README.md)：直接卷积、im2col+GEMM、Winograd 的成本。
3. [Kernel Fusion](03-kernel-fusion/README.md)：减少 launch 与中间数据流量。

## 本阶段原则

先读参考实现和论文，再做简化子问题；记录 shape、dtype、layout、精度和硬件限制；不要把论文性能数字当成自己设备上的承诺。

## 参考资料

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [CUTLASS](https://docs.nvidia.com/cutlass/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
