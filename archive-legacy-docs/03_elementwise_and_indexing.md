# 03：从 Vector Add 学会索引、边界和验证

## 本章目标

能够把一个逐元素公式映射到线程，处理任意长度输入，并用 CPU reference 验证结果。

## 最小问题

```text
C[i] = A[i] + B[i]
```

典型线程索引为 `i = blockIdx.x * blockDim.x + threadIdx.x`。必须用 `if (i < N)` 保护尾部线程，不能假设 `N` 一定整除 block size。

## 小案例

1. Vector Add：验证 1、31、32、33、1000 个元素。
2. ReLU：`Y[i] = max(X[i], 0)`，观察分支和统一计算的差别。
3. 2D Matrix Add：用 `row/col` 索引，验证 row-major 地址 `row * stride + col`。

## 常见错误

- 把 `N` 写成 `N-1`，漏掉最后一个元素。
- 没有边界检查，最后一个 block 越界写回。
- 只比较少量输出，没有检查 NaN、Inf 和完整误差。
- 把线程索引和数据布局混为一谈；`threadIdx.x` 对应哪一维是程序员的映射选择。

## 验证和性能

先用随机输入与 CPU reference 做逐元素比较，再测固定输入的 warmup 和重复运行。Vector Add 通常是 Memory Bound，重点观察有效带宽而不是 FLOPS；若 kernel 很短，还要考虑 launch overhead。

## 本章交付物

`.cu` 源码、CPU reference、边界测试、编译命令、带宽计算和一页性能记录。

## 参考资料

- [CUDA Programming Guide：Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [CUDA C++ Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
