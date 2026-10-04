# 内存层级：算子性能常常取决于数据怎么走

## 1. 从慢而大到快而小

一个简化的 GPU 数据路径：

```text
Host memory ↔ GPU HBM ↔ L2 ↔ L1/shared memory ↔ registers ↔ execution units
```

不同 GPU 的缓存结构、容量和访问规则不同。上图是帮助推理的概念模型，不是每代设备完全相同的硬件图。

- HBM/Global Memory：容量大，访问延迟相对高。
- L2：多个 SM 可共享的缓存层级。
- Shared Memory：程序员管理的 block 内暂存空间，可显式复用数据。
- Registers：线程私有，适合保存索引、局部和 accumulator；过度使用会降低并发或发生 spill。

## 2. 带宽、延迟和吞吐不是同一个概念

- Latency：一次请求要等多久。
- Bandwidth：单位时间可搬多少数据。
- Throughput：单位时间完成多少工作。

GPU 用大量可运行 warp 隐藏单次内存延迟。一个 kernel 可能带宽没有打满，却仍然被延迟、并行度不足或指令发射限制。

## 3. 数据复用例子：矩阵乘法

朴素 GEMM 中，不同输出会重复读取相同的 A 行和 B 列。Tiling 将一小块 A/B 搬入 shared memory，让多个乘加复用：

```text
Global load → shared tile → 多次乘加 → 写回 C
```

复用提高并不免费：shared memory 和寄存器使用增加，可能使每个 SM 能同时驻留的 block 变少。优化是资源和复用之间的权衡。

## 4. 合并访存

当同一 warp 中线程访问相邻地址时，硬件更容易用较少 memory transaction 满足请求。若每线程跨很大步长访问，实际传输字节可能高于程序真正使用的字节。

教学分析应区分：

```text
requested bytes：算法想读取的数据量
transferred bytes：内存系统实际传输的数据量
```

两者差距大时，检查 layout、stride、线程映射和访问对齐。

## 5. Bank conflict

Shared memory 被分成多个 bank。访问模式导致多个线程争用同一 bank 时，访问可能被拆分处理。不要只凭 `[TILE][TILE+1]` 口诀改代码：先推导线程到地址的映射，再看 profiler 是否确认 bank conflict。

## 6. 小练习

1. Vector Add 为什么通常比大 GEMM 更容易受带宽限制？
2. 一个矩阵值被重复使用时，放在寄存器、shared memory 或 cache 的收益和代价分别是什么？
3. `HBM utilization=50%` 能否直接证明 memory bound？还需要什么证据？

## 参考资料

- [CUDA Best Practices · Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-accesses)
- [CUDA Best Practices · Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
