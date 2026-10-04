# Reduction：Sum 和 Max

请先读 [完整手算与代码讲解](walkthrough.md)。本页保留阶段路线和验收要点。

## 为什么它是第一个分水岭

Vector Add 中每个输出独立；Reduce 的输出依赖多个输入。线程先算局部 partial，再进行线程间合并，因此出现同步、竞争、atomic 和浮点加法顺序问题。

## Sum 算法阶梯

本目录的 [block_reduce_sum.cu](block_reduce_sum.cu) 是单 block 教学版，N=1003，包含 CPU reference、shared memory tree reduction 和结果比较。它用于讲解协作，不是任意规模高性能实现。

### V0：一个线程串行求和

正确但并行度低，适合作为 reference，不适合大数组。

### V1：每线程累加多个元素

线程 t 处理 `x[t], x[t+blockDim], ...`，减少单线程循环深度并覆盖全数组。

### V2：block 内 tree reduction

线程写 partial 到 shared memory，同步后按 2 的幂次缩小活跃线程集合。每轮共享数据写完后，其他线程读之前必须有正确的同步。

### V3：多个 block

每个 block 写一个 partial sum，第二个 kernel 再规约 partials。原子操作可以减少阶段，但竞争严重时可能更慢，且浮点 atomic 的累加顺序不确定。

## Max 与 Mean

- Max 的 identity 是负无穷，不能错误地从 0 初始化负数输入的 max。
- Sum 的 identity 是 0。
- Mean = sum / count，count 应是实际有效元素数，尾块的 padding 不能算进分母。

## 典型 bug

1. 少了 barrier，出现读写竞争。
2. 把 grid 同步误当成 block 同步。
3. padding 为 0 后做 max，负数数组结果错误。
4. 对非二次幂长度使用不安全的索引。
5. 用 atomic 得到结果，却要求 bitwise deterministic。

## 验收实验

测试 N=0、1、31、32、33、1023、1024、1025 和大数组；测试全正、全负、交错大数/小数。报告最大绝对误差、有效带宽、block 数和时间。

## 参考资料

- [CUDA Programming Guide · Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUB DeviceReduce](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
