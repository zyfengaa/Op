# 内存层级：算子性能常常取决于数据怎么走

## 1. 从慢而大到快而小

一个简化的 GPU 数据路径：

```text
Host memory ↔ GPU 显存（例如 HBM/GDDR）↔ cache / 显式片上暂存 ↔ registers ↔ execution units
```

不同 GPU 的缓存结构、容量和访问规则不同。上图是帮助推理的概念模型，不是每代设备完全相同的硬件图。

- Global Memory：设备全局地址空间，通常由显存支持；物理显存不一定采用 HBM。
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

## 7. 存储空间、物理存储器和缓存路径不是一回事

global memory 是编程中可访问的设备地址空间；物理显存可能是 HBM，也可能是其他类型，例如 GDDR。因此不能把“GPU 的所有 global memory 都是 HBM”当成定义。一个 global load 可能由 cache 满足，也可能真正访问显存；源码写了一次 load 不等于发生了一次 HBM 事务。

register 通常放线程私有的标量和累加器；shared memory 用来让同一 block 的线程交换/复用数据；local memory 虽叫 local，通常是线程私有地址空间，但可能需要通过设备内存层级访问，常见于寄存器 spill 或某些局部数组。shared 与 local 名字相近，行为却不同。

cache 通常由硬件根据访问模式管理，而 shared memory 的装载、索引和同步由程序显式控制。若硬件 cache 已很好地覆盖复用，搬进 shared 未必值得；若需要明确的 block 内复用与重排，shared 就有价值。

## 8. 合并访存：同时看 32 个线程的地址

用一个简化的现代 CUDA 访存分析模型说明，假设一个 warp 的 32 个活跃线程各读 4 字节，相关请求按 32 字节片段统计。具体事务还受架构和 cache 路径影响。

如果线程 t 读 base+4t，且 base 对齐，32 个线程正好覆盖 128 字节，即 4 个连续的 32 字节片段。如果线程 t 读 base+128t，每个线程落在不同片段，最多需要触及 32 个片段，却仍只使用 128 字节。

| 模式 | 线程真正用到的数据 | 简化模型触及的数据片段 | 有效利用 |
|---|---:|---:|---:|
| 连续 4 字节步长 | 128 B | 4×32 B | 100% |
| 128 字节步长 | 128 B | 32×32 B | 12.5% |

这个表帮助预测浪费，不是实际 profiler 数据。编译指令宽度、未对齐访问、cache 命中和活跃 lane 数都可能改变观测结果。

注意观察对象是“同一条访存指令中，不同线程的地址”，不是只看单线程循环是否顺序。GEMM 的单线程沿 B 的列读取看似跨度大，但相邻线程负责相邻输出列时，在同一轮 k 上访问的 B 地址可以是连续的。

## 9. 用转置理解为什么需要 shared memory 重排

输入 A 的形状 H×W，输出 B 的形状 W×H，B[c,r]=A[r,c]。若相邻线程沿 c 方向处理，读取 A[r,c] 连续，但写 B[c,r] 的地址步长是 H；若换一种映射让写入连续，读取又可能跨大步长。

分块转置的思路是先按输入连续方向协作加载到 shared tile，再让线程按输出连续方向从这个 tile 读取并写出。shared tile 在这里不仅用于数据复用，也作为线程之间交换布局的缓冲。加载和交换之间要有屏障。

这也解释了为什么“每个输入只用一次，所以 shared memory 没用”并不成立。它可能优化的是输出访问组织，而不是重复利用同一个值。

## 10. Bank conflict 的地址演算

假设 shared memory 有 32 个 bank，每个 bank 的基础映射单位为 4 字节，访问为普通 32-bit 标量；以字偏移 w 表示地址，则 bank=w mod 32。这是特定常见条件下的教学模型，向量/更宽访问需重新分析。

对 float tile[32][32]，列 0 中 tile[t][0] 的字偏移是 32t，所以所有线程映射到 bank 0，但访问的是不同地址，可能形成严重冲突。若声明 tile[32][33]，同一列偏移变成 33t，bank=t mod 32，分散到不同 bank。

为什么多加一列有效？因为改变了每行 stride，使列访问不再都落在相同余数。它不是任何 kernel 都应套用的规则：如果本来按行访问就没有这种冲突，加一列可能只是增加存储。多个线程读取同一地址还可能使用广播机制，也不能仅凭“同一个 bank”判定冲突。

## 11. 带宽计算不能漏掉读和写

Vector Add 对 N 个 FP32 元素读 A、读 B、写 C，算法字节数约 12N。如果 N=1,000,000、实测 kernel 时间假设为 0.1 ms，有效带宽为 12,000,000/0.0001=120 GB/s。这里的 0.1 ms 是演算输入，不是本仓库测试成绩。

GEMM 和融合算子的字节计算需先明确中间张量。省掉 N 个 FP32 临时元素的一次写和一次读，理想上少 8N 字节；但如果中间值已有 cache 命中，实测显存字节减少量可能不同。

同一份报告里应明确写“算法有效带宽”还是“硬件计数器测得的显存流量”，以免把不同分母的百分比当成同一指标。

## 12. 资源占用怎样限制并发

假设某个教学设备每 SM 可用于此场景的 shared memory 为 64 KiB，每个 block 使用 20 KiB，那么仅这一资源就把同时驻留数限制在 floor(64/20)=3 个 block。若增加 tile 后每 block 用 36 KiB，就只能放 1 个。真实设备还同时受寄存器、线程数、block 上限和分配粒度约束。

这说明增加局部复用可能减少可用于隐藏等待的并发工作。分析优化时应一起记录“少了多少数据搬运”和“多用了多少资源”，而不是只看某个 tile 的复用次数。

## 参考资料

- [CUDA Best Practices · Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-accesses)
- [CUDA Best Practices · Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
