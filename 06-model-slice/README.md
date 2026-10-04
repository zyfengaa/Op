# 一条模型主线：跑通一个 Decoder Block

这次从模型中的一次前向出发，把前面学到的算子连接起来。你将完成三件事：验证一个完整 block、让 prefill 与带 KV Cache 的 decode 对齐、用实际报告解释局部优化是否改善整体。

它是随机权重的小型推理实验，不是预训练语言模型；没有 tokenizer、词表 head、训练、GQA、dropout、量化或分页 KV Cache。无需下载模型，CPU 版 PyTorch 就能运行。权重和输入固定 seed，参考版和候选版共享同一组权重。

## 先运行，再沿数据流回读

首次接触模型结构，先读 [四通道逐步工作台](walkthrough.md)，运行 `python 06-model-slice/inspect_example.py`。从一行 RMSNorm 的手算、QKV 拆头、RoPE、causal 矩阵一直跟到缓存输出对齐，再进入下面的完整测试和计时。

~~~powershell
python -m unittest discover -s 06-model-slice -p "test_*.py" -v
python 06-model-slice/run.py --profile --output 06-model-slice/results/local
~~~

第一条检查整段与分段、逐 token、causal、RoPE 和 cache 语义；第二条验证结果后分别测 prefill/decode 的两种实现，并保存原始时间样本、shape、环境、实际原生事件、trace、阶段图和 E2E 图。

打开已保存的 [本机 CPU 实测解读](results/cpu-reference/README.md)，与自己的 results/local 对照。这里有实际证据，不要求学生从一张空表开始猜。

## 完整结构：两次归一化、两次残差

~~~mermaid
flowchart TD
  X["X: B×S×D"] --> N1[RMSNorm]
  N1 --> QKV[QKV 投影 GEMM]
  QKV --> R[RoPE：使用绝对位置]
  R --> KV[追加到 KV Cache]
  KV --> A[带 causal mask 的 Attention]
  A --> O[O 投影 GEMM]
  X --> ADD1[残差相加]
  O --> ADD1
  ADD1 --> N2[RMSNorm]
  N2 --> GU[gate / up 投影]
  GU --> ACT["SiLU(gate) × up"]
  ACT --> DOWN[down 投影]
  ADD1 --> ADD2[残差相加]
  DOWN --> ADD2
  ADD2 --> Y["Y: B×S×D"]
~~~

前归一化 block 的 MLP 前还有第二次 RMSNorm，Attention 和 MLP 各有一个残差分支。把这些环节省略，得到的是另一种模型，不应称为同一 block 的优化。

[block.py](block.py) 中的阶段名对应图中的节点，profiler 里以 slice/ 开头。每个阶段都能映射回课程：归约用于 RMSNorm/Softmax，GEMM 用于投影，逐元素用于旋转/激活/残差，在线注意力用于理解 SDPA 的候选后端。

## 工作台一：两种版本，先证明等价

reference 版本分别计算 Q、K、V，显式生成 scores、softmax 概率和 P@V；gate/up 也分别投影。optimized 候选把 QKV 合成一次矩阵乘、gate/up 合成一次矩阵乘，并调用 PyTorch SDPA。这个名字只表示候选，不保证每个尺寸更快。

两个版本共享 packed 权重，reference 使用切片读取同一块权重，因此不会因为参数不同导致比较无效。SDPA 由 PyTorch 按环境分派，不能因为调用了这个 API 就宣称运行了 CUDA FlashAttention。本机原始 trace 中出现的是 CPU 对应路径，详见实测报告。

本版本未手写新的 GPU Attention kernel。它提供的是可运行的模型/算子集成边界，后续你可逐阶段替换后端，并复用整体正确性检查。

## 工作台二：把 prefill 和 decode 的 shape 写出来

默认 B=1、前缀 S=64、D=64、heads=4、head_dim=16、MLP hidden=128：

| 阶段 | prefill | 单 token decode |
|---|---|---|
| 输入 X | [1,64,64] | [1,1,64] |
| packed QKV GEMM | [64,64]@[64,192] | [1,64]@[64,192] |
| Q | [1,4,64,16] | [1,4,1,16] |
| K/V | [1,4,64,16] | [1,4,65,16] |
| 逻辑 scores | [1,4,64,64] | [1,4,1,65] |
| gate/up GEMM | [64,64]@[64,256] | [1,64]@[64,256] |
| down GEMM | [64,128]@[128,64] | [1,128]@[128,64] |

M=B*S，所以 batch>1 的 decode 不是 M=1。权重形状不变，但同一次调用里可复用权重的输入行数不同。读 [decode 小 M GEMM 实验](decode-gemm.md)，再决定是否需要形状特化。

## 工作台三：缓存正确性比“拼接成功”更严格

cache 保存已经做过 RoPE 的 K 和原始 V，形状 [B,H,L,Dh]。新 token 的 position 从已有 L 开始；不能每次从 0 重新旋转，也不能把旧 K 再旋转一次。位置编码采用相邻元素成对的约定，不与采用 split-half 约定的 checkpoint 直接混用。

causal mask 按绝对 query 位置比较 key 索引。decode 的局部 query 下标为 0，但绝对位置是 L；若把矩形 attention 的 mask 简单当作左上三角，就可能只看见第一个 key。实现显式构造允许访问的布尔 mask，并传给 SDPA。

当前 cache 使用 torch.cat 形成新张量，便于解释且不原地修改旧 cache；这包含额外拷贝成本。它不是生产系统的预分配/分页缓存，计时范围包含这个成本，不能直接推断真实推理框架的 cache 性能。

## 可判定验收

test_block.py 的 7 项检查覆盖：

1. packed/SDPA 与显式路径的数值一致。
2. 先处理前缀再处理后缀，与完整 causal 前向的后缀一致。
3. 每次一个 token 的输出拼接，与完整输出一致。
4. 改变未来 token，不影响过去输出。
5. RoPE 保持每对元素的平方和，并使用实际位置。
6. 常量输入的 RMSNorm 不等同于减均值的 LayerNorm。
7. 非法配置、空输入和错误 cache 被拒绝。

测试使用 float64 小尺寸进行严格对照；性能脚本默认 float32，再执行相应容差的前向检查。它们不验证训练梯度，因为本实验是推理切片。

## 工作台四：E2E、阶段 profile、内存各回答什么

普通计时在 inputs/weights 已就绪后包围完整 block，CUDA 时同步，包含临时分配和 cache 拼接；不含模型构造、分词、主机传输或词表生成。每次 decode 计时都使用相同的不可变前缀 cache，防止越测越长。

profile 另跑一次，不与微基准混合。阶段占比来自同一次采集中的非重叠 CPU 范围；不能把 instrumented CPU 占比当成 GPU SM 利用率，也不能把各层嵌套事件的 total 时间直接相加。

CUDA 环境会额外记录 allocator 的 baseline、peak allocated 与相对增加量；CPU 环境的显存字段为 null，并记录可手算的 logical KV 字节数。逻辑字节数不是 RSS 或峰值显存。本仓本机报告没有 GPU 显存成绩，未来有设备时直接重跑：

~~~powershell
python 06-model-slice/run.py --device cuda --profile --output 06-model-slice/results/my-gpu
~~~

## 三项模型级挑战

**挑战 A：只改一个阶段。** 先保留 reference Attention，只合并 QKV；或只切换 SDPA，保持其他投影。当前候选同时改变多个阶段，不能直接将 E2E 差异全部归因于 FlashAttention。

**挑战 B：改变形状。** 用 --seq 128 --width 256 --heads 4 --hidden 512 跑同样验收。解释 M/K/N 及 scores 规模如何变化，先写预测再测量；当前默认 tiny block 不代表大模型硬件瓶颈。

**挑战 C：故意破坏缓存。** 在本地练习分支把 RoPE position 重置为 0，或把 mask 改成不带绝对偏移。运行逐 token 测试定位错误，修好后再提交。不要仅用“一次 decode 没有报错”作为通过标准。

## 参考资料

- [RoFormer](https://arxiv.org/abs/2104.09864)：旋转位置编码与位置关系。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：mask、dropout 与后端分派。
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)：事件、shape 与内存记录。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：理解避免完整分数矩阵写回的思想。
