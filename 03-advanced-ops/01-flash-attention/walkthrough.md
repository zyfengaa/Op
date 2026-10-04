# Attention 学习实验：先证明语义，再理解分块

本章先学习简化的 scaled dot-product attention，不把这个 CPU 案例称为 FlashAttention 实现。运行 [advanced_lab.py](../../examples/cpu/advanced_lab.py) 可验证小矩阵；真正的 GPU 高性能实现还需处理内存层级、流水线和设备细节。

## 1. 可手算的两个 token

令 d=1，Q=[[1],[0]]，K=[[1],[0]]，V=[[10],[20]]。QKᵀ=[[1,0],[0,0]]。第一行 softmax([1,0])≈[0.731059,0.268941]，所以 O[0]≈0.731059×10+0.268941×20=12.68941；第二行权重 [0.5,0.5]，O[1]=15。因 d=1，缩放系数 1/sqrt(d)=1；d>1 时不能漏掉该缩放。

Causal mask 时，第 0 个 query 只能看到第 0 个 key，因此 O[0]=10；第 1 个 query 可见两个 key，O[1]=15。注意 mask 应在 softmax 前作用于 logits，而不是算出概率后随意清零却不重归一化。全屏蔽行没有自然的 softmax 分母，须制定明确策略。

## 2. 为什么需要分块

显式 S=QKᵀ 和 P=softmax(S) 对长度 L 分别有 L² 个元素。若每元素 4 字节，单个 L=4096 的矩阵约 64 MiB；同时保留 S/P 会增加中间存储和流量。这只是数量级计算，不是任何具体模型的内存实测。分块做 QKᵀ、局部 softmax 统计和 P·V，可避免把整张 P 写到全局内存。

对某 query 的旧块，维护 m、l 和未归一化向量 o=Σ exp(score-m)*V。新块有 m_b、l_b、o_b，令 m'=max(m,m_b)，则：

~~~text
l' = exp(m-m')*l + exp(m_b-m')*l_b
o' = exp(m-m')*o + exp(m_b-m')*o_b
最终 O = o'/l'
~~~

和前章 online softmax 一样，旧统计必须按新最大值重缩放；只更新 l 而忘记 o，会使输出错。小样本可按两个 key 分成两块手算核验。

## 3. 从 reference 到真实 kernel 的阶段

1. 用 CPU 显式 S/P/O reference 固定 shape、mask、scale 和 dtype。
2. 在 GPU 上做三个独立 kernel，与 CPU 数值对比。
3. 为 QKᵀ 分 tile，检查越界 K/Q 位置的 mask。
4. 加 online m/l/o，先无 dropout、无变长、无 GQA。
5. 融合读取 K/V tile，测量峰值显存和端到端时间。
6. 最后扩展 causal、变长、GQA、训练反向；每加一种语义都新增测试。

不要把这里的简化 recurrence 直接当成生产 kernel：warp 分工、寄存器上限、shared memory、异步拷贝、精度和反向传播都尚未解决。

## 4. 验证实验与排错

~~~powershell
python examples/cpu/advanced_lab.py
python -m unittest discover -s examples/cpu
~~~

修改 Q/K/V 为相同值，观察权重均匀；修改为极大点积，检查稳定 softmax；测试 L=1、L=tile±1、causal 与非 causal。输出需同时检查数值误差和 mask 语义。GPU 性能报告要分 prefill（较长 Q）与 decode（Q 长度 1），不要混为一谈；同时记录 L、head_dim、dtype、batch、mask 和设备。

## 练习与答案

练习：上述两 token 例子中 causal 模式的 O 是什么？答案 [[10],[15]]。若对第 0 行先做完整 softmax 再把第二个概率置零，输出会变成约 7.31059，说明漏了归一化。

## 参考资料

1. [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：IO-aware 思路和在线归约。
2. [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)：阅读真实内核时对照接口/版本。
3. [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：分块映射示例。
