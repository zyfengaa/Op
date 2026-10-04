# GEMM V3：向量化 Load/Store

本目录是接在可运行 V2 后的设计实验，尚未提供经过设备验证的向量化 GEMM kernel。先完成下列地址和尾部分析，再实现候选版；不能把本目录名字当成 V3 性能已经完成的证据。

## 学习目标

理解更宽的每线程 load 可能减少指令数或更有效利用内存 transaction，但它依赖地址对齐、连续访问、尾部 mask 和编译器生成代码。

## 前置条件

- V2 的访存和正确性已理解。
- 指针与每行 stride 的对齐情况已知。
- 当前设备支持目标向量宽度。
- 尾部元素有 scalar/masked fallback。

## 实验方法

对比标量 load 与向量化 load，固定 shape、tile、编译 flags 和频率条件。检查生成代码/指令、transaction、带宽和绝对 kernel latency。若只有规整 shape 更快，应采用 shape-specialized dispatch 或明确限制支持范围。

## 常见失败

- 假设 base pointer 对齐但实际切片偏移。
- 跨越行尾读取下一行数据。
- 向量化后寄存器压力增加，occupancy 下降。
- 数据量太小，测量噪声大于差异。

## 用 K=41 证明“分配对齐”不代表“每一行对齐”

FP32 A 每行 41 个元素，一行 164 字节。即使 base 是 16 字节对齐，row 0/1/2/3 的起始地址模 16 分别为 0、4、8、12，只有 row 4 又回到 0。每行都从 col=0 开始强制 float4 读取，会遇到未满足自然对齐的地址。

K=40 则每行 160 字节，行起点保持 16 字节对齐，但仍要保证当前 col 是四个元素的对齐位置。若输入是从大矩阵第 1 列开始的 view，base 本身又偏移了 4 字节。由此可见，对齐条件依赖 base、stride 和列偏移三项。

## 线程分工会随向量宽度变化

标量版每线程搬一个数；每线程搬四个数后，原来的线程总数和分配公式不能保持不变，否则会重复搬运或覆盖。要重新计算 tile 中需要多少个四元素片段，每个线程负责哪些片段，以及最后不足四个元素如何回退。

例如一行 10 个数，完整向量片覆盖 [0..3] 与 [4..7]，尾部只剩 [8,9]。若对尾部仍进行四元素读取，读出 [10,11] 即越界，即使最后写回有 mask 也救不了前面的非法读取。

## 为什么 wider load 不保证更少的显存事务

32 个线程各读一个连续 float，本来就可能有效合并为少量事务。改为每线程读取 float4，首先改变的是每线程指令与任务粒度；实际事务数取决于总覆盖范围和对齐，不能把指令数降低直接等同于 HBM 字节降低。

比较时保持数学工作量和测量范围相同，检查编译产物是否真的产生目标访存宽度，再看吞吐、寄存器和实际 latency。若 compiler 已自动向量化原版本，手工改写可能只是增加复杂度。

## 验收练习

列出 K=40、41，base 偏移 0、1 个 float，以及 N 为 tile-1/tile/tile+1 的组合。对每一组写出向量路径的合法区间和 scalar fallback 区间，再运行 correctness 和内存检查工具。最后记录向量化适用条件，作为 dispatch 判断，而不是给所有 shape 强行使用一个版本。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：查 global memory 的大小与自然对齐要求。
- [CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)：区分跨线程合并与单线程向量访问。
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)：进阶阅读 tile 搬运、布局和向量访问约束。
