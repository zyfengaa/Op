# GEMM：从 baseline 到瓶颈分析

先读 [GEMM 完整案例](walkthrough.md)：从具体矩阵乘积到 tile 复用、边界、性能证据。decode 阶段的小 M 情形单独见 [decode GEMM](decode-gemm.md)——同一个 matmul API 不等于同一个优化问题。

## 问题

C[M,N] = A[M,K] × B[K,N]，计算量约为 2MNK。接口必须声明 A/B/C 的 shape、dtype、layout、stride 和累加类型。

## 本仓已有的实现

| 版本 | 文件 | 作用 |
|---|---|---|
| V1 Naive | [naive_gemm.cu](impl/v1-naive/naive_gemm.cu) | 每线程一个 C 元素，建立 correctness 与 baseline |
| V2 Tiled | [tiled_gemm.cu](impl/v2-tiled/tiled_gemm.cu) | block 把 A/B tile 搬到 shared memory，提升复用 |
| CPU 教学实验 | [gemm_lab.py](../../assets/cpu/gemm_lab.py) | 无 GPU 时对照 naive/tiled 的索引与数据流 |

V1 和 V2 都内置小尺寸 CPU reference 比较。V2 用 TILE=16，刻意测试 M=37、N=29、K=41 的尾块——三个维度都不整除。该正确性小例不代表大矩阵性能，也不替代 benchmark。

**还没有的**：V3 向量化、register tiling、MMA、double buffer。这些属于 [04-performance](../../04-performance/README.md) 的手法，本文档只解释它们改变哪一层，不声称仓库已有实现。

## Roofline

理论 Arithmetic Intensity 要结合实际 traffic 解释。大 GEMM 常有较高复用；decode 的 M=1 GEMM 可能每个权重只读一次。算清上界的方法见 [性能分析](../../04-performance/analysis.md)。

## 每个版本的分析记录

写明设备、shape、编译器、时间、GFLOPS、理论与实测 traffic、occupancy、主要 stall、假设和是否保留。数据没有实测时标为模拟数据。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

## 相关页面

- [matmul walkthrough](walkthrough.md)：tile 装载、屏障、地址公式的手算推导
- [decode-gemm](decode-gemm.md)：小 M 的权重复用与算术强度
- [手法 · Tensor Core](../../04-performance/techniques/tensor-core.md)：矩阵指令改变了哪一层
- [手法 · 访存模式](../../04-performance/techniques/memory-access.md)：合并访存与向量化
- [常见错误清单](../../99-reference/common-pitfalls.md)：GM01 协作装载与屏障
