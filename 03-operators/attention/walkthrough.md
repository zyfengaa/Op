# Attention 学习实验：先证明语义，再理解分块

本章先学习简化的 scaled dot-product attention，不把这个 CPU 案例称为 FlashAttention 实现。运行 [advanced_lab.py](../../assets/cpu/advanced_lab.py) 可验证小矩阵；真正的 GPU 高性能实现还需处理内存层级、流水线和设备细节。

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
python assets/cpu/advanced_lab.py
python -m unittest discover -s assets/cpu
~~~

修改 Q/K/V 为相同值，观察权重均匀；修改为极大点积，检查稳定 softmax；测试 L=1、L=tile±1、causal 与非 causal。输出需同时检查数值误差和 mask 语义。GPU 性能报告要分 prefill（较长 Q）与 decode（Q 长度 1），不要混为一谈；同时记录 L、head_dim、dtype、batch、mask 和设备。

## 5. 练习与答案

练习：上述两 token 例子中 causal 模式的 O 是什么？答案 [[10],[15]]。若对第 0 行先做完整 softmax 再把第二个概率置零，输出会变成约 7.31059，说明漏了归一化。

## 6. 为什么只存一个归一化输出还不够

假设第一块只有 score=0、V=10，它的归一化输出是 10。第二块只有 score=2、V=20，它的归一化输出是 20。如果把两个输出直接平均得到 15，就错误地认为两个块总权重相等；事实上第二块的指数权重大约是第一块的 7.389 倍。

所以状态不只需要“当前输出”，还需要当前最大值 m、分母 l，以及未归一化的加权和 o。它们一起保留足够信息，以便新块改变最大值时重新解释旧块权重。

逐块状态如下：

| 步骤 | m | l | o | 当前 o/l |
|---|---:|---:|---:|---:|
| 处理 score=0、V=10 | 0 | 1 | 10 | 10 |
| 新块最大值变为 2 | 2 | exp(-2)+1 | 10exp(-2)+20 | 18.807971 |

第二步的 alpha=exp(0-2)≈0.135335 同时乘旧 l 和旧 o。只缩放 l、不缩放 o，将得到 30/1.135335≈26.4239，甚至超出 V 的 [10,20] 范围。对有限值的一行，softmax 权重非负且和为 1，所以每个输出分量应在该行可见 V 对应分量的最小值与最大值之间；这是很实用的额外检查。

## 7. 真正运行一个分块在线版本

[blocked_attention](../../assets/cpu/dataflow_lab.py) 与 [显式 attention reference](../../assets/cpu/advanced_lab.py) 是两个独立计算路径。前者逐 K/V 块维护 m/l/o，后者先得到一行所有 logits 和权重。运行：

~~~powershell
python assets/cpu/dataflow_lab.py --topic attention
python -m unittest discover -s assets/cpu -v
~~~

数据流实验使用 Q=[[1]]、K=[[0],[2]]、V=[[10],[20]]，block_size=1，正好对应上表。它会打印每个块结束后的 m/l/o。单元测试进一步使用多 query、多特征、非整除块长与 causal mask，把两个路径逐元素比较。

代码中先计算 new_max，再计算 alpha，随后使用相对于 new_max 的新块 weights：

~~~python
alpha = exp(old_max - new_max)
new_total = alpha * old_total + sum(weights)
new_acc = alpha * old_acc + sum(weights[j] * V[j])
~~~

初始 total=0、m=-∞，本实现明确令旧贡献系数为 0。全屏蔽的块先跳过，不让 -∞-(-∞) 生成 NaN。最终若整行没有任何可见 key，则报错，因为本实验没有定义全屏蔽行输出策略。

## 8. mask 的坐标与 KV Cache 的位置

本实验 causal 采用左上对齐：第 qi 行只允许 key 下标 kj<=qi。Q 与 K 长度相等的自注意力容易理解这一点；但 decode 时 Q 可能只有最后一个 token，K 包含全部历史。此时 query 的局部索引 0 不等于绝对序列位置 0，直接套 kj<=qi 会错误地只允许第一个 key。

带 KV Cache 的实现必须明确 query 的绝对起始位置、历史长度和 mask 对齐约定。新增参数 q_offset 后，某些场景会使用 kj<=q_offset+qi，但还要与目标框架/缓存协议一致。本仓库小实验没有 KV Cache 参数，不能据此声称已经支持 decode。

padding mask、causal mask 和 tile 越界 mask 也有不同含义：前者表示无效 token，中者表示不可见的未来 token，后者表示实现为了凑 tile 生成的不存在位置。它们最终都影响 logits/贡献，但来源应分开，方便检查。

## 9. GPU 分块需要规划哪些活跃数据

对一个 Q block，通常需要当前 Q 片、某个 K/V 片、score/probability 的局部片，以及每个 query 的 m/l/o。head_dim 增大时，o 的向量长度增长；query tile 增大时，统计状态和局部分数片也增长。

这解释了为何“把 tile 加大减少访问”会遇到资源限制。score tile 的元素数近似 Br×Bc，输出累加器约 Br×Dv；它们可能分布在寄存器和 shared memory 等位置，实际布局取决于实现。不能仅凭完整 L×L 矩阵不落盘就忽略片上资源。

分块 CPU 实验只验证数学上的状态合并，不模拟 warp 布局、矩阵指令、bank conflict 或异步流水线。要转成 GPU kernel，先用 GEMM 章节的方法证明每个 tile 的加载与输出位置，然后再优化搬运和计算的重叠。

## 10. 反向、dropout 与 GQA 增加了哪些约束

反向传播可能通过保存统计量和重算局部概率降低中间存储，但多做计算；必须明确保存哪些前向状态。dropout 要处理随机数状态和可复现性，重新计算时不能换一套随机 mask。GQA 则让多个 query head 共享较少的 key/value head，增加 head 索引映射与复用设计。

这些都不是在已有 kernel 后加一个布尔参数就自动完成的。每个功能分别改变数学契约或数据流，需要新的 reference、边界输入和集成测试。阅读生产实现时，先找到对应分支，再跟踪其索引和状态。

## 11. 本章作业与答案要点

把两 token 例子的 block_size 从 1 改到 2，输出仍应约 18.807971。把 V 同时加 5，输出也应加 5，因为权重和为 1。把 Q/K 改到使新块最大值大幅上升，检查旧 o 与 l 都按同一个 alpha 缩放。

如果结果超出可见 V 的逐分量范围，先查分母、mask 和 o 的重缩放；但这条范围检查不能代替完整数值对照。提交至少一段状态 trace 和一个故意漏掉 alpha 的失败反例。

## 参考资料

1. [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：IO-aware 思路和在线归约。
2. [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)：阅读真实内核时对照接口/版本。
3. [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：分块映射示例。
