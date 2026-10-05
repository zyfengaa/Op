# 量化域

量化把 fp32（或更高）的权重、激活压成 int8、fp8、int4 等低比特表示，用更少的字节和更快的矩阵单元换取吞吐。它已经是当前推理算子开发的主要工作量。这个域回答**量化本身**的问题：数值怎么映射、kernel 怎么写、精度代价怎么量。

## 这个域回答什么

| 问题 | 页面 |
|---|---|
| 低精度格式到底能表示哪些数，为什么 bf16 和 fp16 不能互换 | [低精度基础](basics.md) |
| 权重和激活怎么从 fp32 映射到整数，scale 从哪来 | [量化算法](algorithms.md) |
| 反量化放在 kernel 的哪一步，累加用什么 dtype | [量化 kernel](kernels.md) |
| 怎么证明量化之后模型没坏，精度代价有多大 | [精度评估](evaluation.md) |

## 与其他域的分工

量化不是独立的数学分支，它落在已有的分工上：

| 域 | 负责 | 与量化的关系 |
|---|---|---|
| [02-foundations/contracts](../02-foundations/contracts/dtype-and-precision.md) | dtype 的语义、参考值口径、容差原则 | 量化是“把 dtype 换成更低比特”，但**参考值必须用真正传进 kernel 的低精度输入算**这一条不变 |
| [03-operators](../03-operators/README.md) | 算子本身的语义与实现 | 量化 GEMM 的 tiling、epilogue 结构沿用 [matmul](../03-operators/matmul/README.md)，只是数据类型和 epilogue 内容变了 |
| [04-performance/techniques/fusion.md](../04-performance/techniques/fusion.md) | 融合的通用判据 | 把反量化融进 epilogue 属于融合，判据（是否省下全量内存往返）通用 |
| **07-quantization（本域）** | 量化的算法、kernel 与评估 | 上面三者回答“怎么算得快”，本域回答“用低比特还能不能算对” |

一句话：**精度基础讲的是 dtype 语义，算子域讲的是算子实现，本域讲的是把二者接通的那层量化映射、它的 kernel 形态和它的代价。**

## 导航

| 页面 | 内容 | 读完能做什么 |
|---|---|---|
| [basics.md](basics.md) | fp16 / bf16 / fp8 的位宽与动态范围、混合精度、累加器精度 | 判断某个张量该用哪种低精度，能不能溢出 |
| [algorithms.md](algorithms.md) | 仿射量化、对称/非对称、粒度、PTQ/QAT、GPTQ/AWQ/SmoothQuant 定位 | 选定量化方案和粒度，知道 scale 从哪来 |
| [kernels.md](kernels.md) | 反量化时机三条路线、int8 累加、per-group scale 的访存、融合顺序 | 设计量化 kernel 的数据路径并避开常见错位 |
| [evaluation.md](evaluation.md) | 单算子误差、端到端指标、参考值口径、逐层敏感度 | 给量化结果出具可信的精度-收益报告 |

## 两条使用路径

**学（从头建立）**：按 [basics](basics.md) → [algorithms](algorithms.md) → [kernels](kernels.md) → [evaluation](evaluation.md) 顺序读。前三页建立“能写”，第四页建立“能证明写对了”。

**查（带着具体问题）**：

| 症状 | 直接翻到 |
|---|---|
| 我的 fp16 累加溢出成 `inf` | [basics.md](basics.md) 的溢出一节 |
| 不确定该用对称还是非对称、per-tensor 还是 per-group | [algorithms.md](algorithms.md) 的粒度对照表 |
| kernel 写完误差总是比预期大 | [kernels.md](kernels.md) 的融合顺序一节 |
| 量化后困惑度掉了，想定位是哪一层的锅 | [evaluation.md](evaluation.md) 的逐层敏感度分析 |

## 一条边界声明

**量化一定带来精度损失，必须用实际数据评估，不能只看理论误差上界。**

理论（例如“均匀量化最大绝对误差是半个量化步长”）给的是上界，它假设输入已被约束在量化范围内、且误差独立。真实激活的分布有长尾、有离群通道、有跨层耦合，误差在层间会放大或抵消。所以本域的任何精度结论，都要求一条实际数据证据；只给理论界的结论视为不完整。具体口径见 [evaluation.md](evaluation.md)，与 [dtype-and-precision.md](../02-foundations/contracts/dtype-and-precision.md) 的参考值原则一致。

## 相关页面

- [术语表](../99-reference/glossary.md)：scale、zero-point、PTQ/QAT 等词条
- [测量口径](../04-performance/measurement.md)：量化收益要按哪种计时范围报
- [验收阶段](../01-workflow/08-verify.md)：正确性与性能两份证据的要求
