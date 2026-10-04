# 04：Reduce 与 Softmax：从同步到数值稳定

## 本章目标

理解一个输出依赖多个输入时，为什么需要分阶段归约、shared memory、同步和数值稳定处理。

## Reduce

以求和为例：`y = sum(x[i])`。单线程串行正确但慢；并行版本先让线程局部累加，再在 block 内做 tree reduction，最后由多个 block 的部分和完成第二阶段归约。

必须明确：block 内 `__syncthreads()` 只能同步同一个 block，不能直接同步全 grid。跨 block 归约可以使用第二个 kernel、原子操作或 cooperative launch，但每种方案都有代价。

## Softmax

直接计算 `exp(x[i]) / sum(exp(x))` 可能溢出。稳定形式为：

```text
m = max(x)
sum = Σ exp(x[i] - m)
y[i] = exp(x[i] - m) / sum
```

教学案例按“求 max → 求 sum → 写回”实现，再讨论 fused softmax 如何减少中间结果的 Global Memory 流量。

## 小案例

1. Block Reduce Sum：比较串行、shared-memory tree、warp shuffle。
2. Row-wise Softmax：每行独立归约，测试不同列数和非 32 对齐尺寸。
3. Fused Softmax：在寄存器/shared memory 中保留中间值，比较读写次数。

## 常见错误

- 只同步读取，不同步写入 shared memory。
- 用 `exp(x)` 而不减去 max。
- 把 block 内同步当成 grid 全局同步。
- 对空行、很长行和含有 Inf/NaN 的输入没有定义行为。

## 性能验证

Reduce 通常同时受访存、归约同步和并行度影响。记录每个阶段的 Global Memory traffic、warp stall、有效带宽和吞吐；不要只看最终 kernel 时间。

## 参考资料

- [CUDA Programming Guide：Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Triton Tutorials：Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/)
