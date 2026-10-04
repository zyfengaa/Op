# FlashAttention：把前面学过的内容组合起来

## 为什么它是进阶项目

Attention 将矩阵乘法、softmax、数据分块、数值稳定、融合、寄存器/shared-memory 规划和流水线组合在一起。若还不理解 Reduce、Softmax 和 GEMM tile，直接读生产实现会被大量细节淹没。

## 数学语义

对 Q、K、V：

~~~text
S = Q × Kᵀ / sqrt(d)
P = softmax(S)
O = P × V
~~~

普通实现会 materialize 中间矩阵 S 和 P。序列长度为 L 时，空间随 L² 增长，且中间结果需要写入和重新读取 HBM。

## Online softmax 的核心直觉

按 K/V 分块处理。对每个 query block 维护 running maximum m、归一化因子 l 和加权输出 accumulator。遇到新块时重缩放旧 accumulator，再合并新块贡献。这样可避免把完整注意力矩阵落到 HBM。

这是算法直觉，不是可直接复制的完整 kernel；mask、causal、dropout、GQA、变长序列和精度策略都需要单独定义。

## 由易到难的实验

1. CPU reference：小矩阵、显式 S/P。
2. GPU 分阶段 attention：验证精度和 mask 语义。
3. tiled QKᵀ：检查 K tile 复用。
4. online softmax：只针对无 dropout 的简化语义实现。
5. fused output：避免 P 中间矩阵 materialization。
6. 加 causal mask、变长 batch，再做性能调优。

## 测试清单

- sequence length 1、非 tile 整除和长序列。
- causal 与 non-causal。
- FP32 reference 对 FP16/BF16 输出。
- 极端 logits 和 mask 全屏蔽边界。
- 输出误差、峰值显存、kernel 时间和 E2E。

## 阅读代码时的问题

- 每个 tile 的 Q/K/V 各从哪级内存来？
- running max 和 running sum 如何重标定？
- 哪些中间张量没有写回 HBM？
- tile 变大对复用、寄存器和 occupancy 有何影响？
- 当前优化针对哪代 GPU 的哪些硬件能力？

## 参考资料

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention reference implementation](https://github.com/Dao-AILab/flash-attention)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
