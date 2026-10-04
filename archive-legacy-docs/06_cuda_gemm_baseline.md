# 06：CUDA GEMM Baseline

## 本章目标

从 CPU reference 和 naive CUDA GEMM 建立可重复 baseline，知道优化前究竟慢在哪里。

## 接口

```text
C[M,N] = A[M,K] × B[K,N]
A、B、C 为 row-major；累加使用 float；输入可以是 float 或后续扩展为 half。
```

## 三个版本

1. CPU reference：最清晰的三重循环。
2. Naive CUDA：一个 thread 计算一个 C 元素。
3. Tiled CUDA：block 协作搬运 A/B tile 到 shared memory。

## Baseline 记录表

| 字段 | 示例 |
|---|---|
| Shape | M/N/K |
| dtype | FP32/FP16 |
| warmup/iterations | 预热与重复次数 |
| correctness | max abs / relative error |
| latency | 平均值与分位数 |
| throughput | GFLOPS |
| traffic | 理论 bytes 与实际指标 |

GFLOPS 计算：`2 × M × N × K / seconds / 1e9`。不要把 Host-to-Device 拷贝时间混入 kernel benchmark，除非目标是端到端应用。

## 边界策略

grid 使用 ceil division；每个线程在读取 A/B 和写回 C 时都进行边界判断。越界 tile 读取可以填零，不能让未初始化 shared memory 参与计算。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA C++ Best Practices：Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
