# Softmax：Reduction、数值稳定和融合

先读 [Softmax 完整推导](walkthrough.md)：真实数字、online 合并公式、mask 边界和实验步骤。

## 定义

对一行输入 x，yᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)。每个输出依赖整行，因此不是普通逐元素 kernel。

## 数值稳定

大正数可能使 exp(x) overflow。利用 softmax 对平移不变的性质：

~~~text
m = max(x)
yᵢ = exp(xᵢ - m) / Σⱼ exp(xⱼ - m)
~~~

计算结果数学等价，指数的输入范围更安全。max 和 sum 的 accumulator dtype 应按平台与框架精度要求决定。

## V0：多 kernel 版本

1. Kernel A 求每行 max。
2. Kernel B 求 exp(x - m) 的每行 sum。
3. Kernel C 归一化写 y。

步骤清楚，适合学习和排错；缺点是多次 launch 和中间 tensor 读写。

## V1：一行一个 block

一个 block 处理一行，依次做 block reduction max、sum，最后写结果。适合行长度和资源相匹配的场景。极长行可能要分块；大量极短行则可能因每行一个 block 而浪费并行资源。

## V2：Online Softmax 思路

分块扫描时维护当前最大值 m 和归一化和 l。每读入新块，重缩放旧和与新和，使算法不必把全部 exp 中间值写回 HBM。实现要从算法推导，不要直接复制不完整伪代码。

## Profile 要看什么

- launch 数量和端到端时间。
- Global Memory 读写量是否因融合下降。
- row 长度与 CTA/warp 工作量。
- reduction 同步、寄存器压力和 occupancy。
- 输出误差、极端输入和非对齐尾块。

## 常见错误

- 忘记减 max。
- max 的无效 lane 用 0 填充，失去稳定性保证；如极负输入会因指数下溢而失败，详见 walkthrough 的反例。
- sum 阶段把 padding lane 的贡献错误计入。
- 要求低精度结果与 FP32 reference bitwise 相等。

## 练习

实现清晰的多阶段版本，再设计 fused row kernel；测试 row 长度 16、128、1024、8192。每种尺寸都说明映射是否合适，并分别测 kernel 与整体调用时间。

## 参考资料

- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
