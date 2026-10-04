# GEMM：从 baseline 到瓶颈分析

先读 [GEMM 完整案例](walkthrough.md)：从具体矩阵乘积到 tile 复用、边界、性能证据。

## 问题

C[M,N] = A[M,K] × B[K,N]，计算量约为 2MNK。接口必须声明 A/B/C 的 shape、dtype、layout、stride 和累加类型。

## 版本路线

- [V1 Naive](v1-naive/README.md)：每线程一个 C 元素，建立 correctness 和 baseline。
- V2 Tiled：block 将 A/B tile 搬到 shared memory，提升复用。
- V3 Vectorized：确认对齐和连续访问后再尝试宽 load/store。
- 后续：register tiling、MMA、double buffer；见 03-advanced-ops 和 references。

## Roofline

理论 Arithmetic Intensity 要结合实际 traffic 解释。大 GEMM 常有较高复用；Decode 的 M=1 GEMM 可能每个权重只读一次。相同 matmul API 不代表相同优化问题。

## 每个版本的分析记录

写明设备、shape、编译器、时间、GFLOPS、理论与实测 traffic、occupancy、主要 stall、假设和是否保留。数据没有实测时标为模拟数据。

V1 和 V2 的最小 CUDA 程序都内置小尺寸 CPU reference 比较。V2 用 TILE=16，刻意测试 M=37、N=29、K=41 的尾块。该正确性小例不代表大矩阵性能，也不替代 benchmark。

## 参考资料

- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
