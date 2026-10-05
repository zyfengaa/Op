# 术语表

按主题分组。每个词条给一句话定义，加一个“为什么重要”。更深的展开在各知识域页面。

## 执行模型

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **SIMT** | Single Instruction, Multiple Threads。一组线程执行同一条指令，各自持有不同的数据 | 是理解分支发散和锁步执行的前提 |
| **warp** | 硬件调度的最小单位，32 个线程锁步执行 | 分支发散、shuffle、coalescing 都以 warp 为单位 |
| **block** | 由多个 warp 组成，可通过 shared memory 和 `__syncthreads()` 协作 | 是能共享数据的最大范围 |
| **grid** | 由多个 block 组成，block 之间默认无法直接同步 | 跨 block 协作需要两阶段或原子操作 |
| **SM** | Streaming Multiprocessor，物理执行单元，同时驻留多个 block | occupancy 的分母 |
| **occupancy** | 一个 SM 上实际驻留的 warp 数与硬件上限之比 | 隐藏延迟的手段，**不是目标本身** |
| **分支发散** | 同一 warp 内的线程走了不同分支，硬件只能串行执行两条路径 | 性能下降的常见原因 |
| **锁步** | 同一 warp 的线程在同一时刻执行同一条指令 | 理解 `__syncthreads()` 和 divergence 的基础 |

## 内存

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **全局内存** | 设备上的大容量内存（HBM/GDDR），所有线程可见 | 主要瓶颈来源 |
| **shared memory** | block 内共享的片上高速存储 | 复用数据、降低全局访存 |
| **寄存器** | 每线程私有的最快存储 | 数量决定 occupancy |
| **local memory** | 寄存器溢出时的后备，实际在全局内存上 | 溢出会静默拖慢 |
| **合并访存（coalescing）** | 同一 warp 的访问落入尽量少的连续内存段 | 决定有效带宽的第一因素 |
| **bank conflict** | shared memory 中同一 warp 的多个线程访问同一 bank 的不同地址，被迫串行 | 用 padding 消除 |
| **L2** | 多 SM 共享的片上缓存 | 跨 block 数据复用的场所 |
| **stride** | 张量相邻元素在内存中的间隔 | stride 不等于 1 表示非连续 |
| **contiguous** | 内存布局与逻辑形状一致，stride 为 1 | 向量化和合并访存的前提 |
| **对齐（alignment）** | 地址是某个字节数的整数倍（如 16 字节） | 向量化加载的硬性要求 |

## 性能分析

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **算术强度 AI** | FLOPs / 搬运字节数 | 判断内存受限还是计算受限 |
| **Roofline** | 用 AI 和机器平衡点给出的性能上界 | 判断“还有多少空间” |
| **机器平衡点** | 峰值算力 / 峰值带宽，单位 FLOPs/byte | AI 与它比较决定瓶颈类型 |
| **有效带宽** | 实际搬运字节数 / 时间 | 与峰值对比得到利用率 |
| **stall** | warp 因等待资源无法发射指令 | stall 原因是定位瓶颈的关键证据 |
| **long scoreboard** | 等全局访存返回的 stall | 指向访存延迟 |
| **MIO throttle** | shared memory 或特殊函数端口饱和 | 指向片上资源 |
| **compute bound** | 受算力限制 | 此时优化访存无用 |
| **memory bound** | 受带宽限制 | 此时优化计算无用 |
| **latency bound** | 并行度不足以隐藏延迟 | 提高 occupancy 才有用 |
| **Amdahl 定律** | 整体提速上限由未优化部分决定 | 决定“值不值得做” |
| **ILP** | 指令级并行，同一线程内多条独立指令重叠执行 | 提高吞吐的另一种手段 |
| **warmup** | 正式采样前的空跑，排除首次调用的额外开销 | 没有它，数字不可比 |
| **计时范围** | 计时包含哪些环节（kernel-only / op-level / E2E） | 不同范围的数字不可比 |

## 归约与同步

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **归约（reduction）** | 把一组值合并成一个（sum / max / min） | 最常见的一类算子 |
| **单位元（identity）** | 与任何值运算都不改变它的元素：sum 是 0，max 是 `-inf` | padding 和越界线程必须贡献单位元 |
| **树形归约** | 逐层两两合并，深度为 log N | 比串行归约快且误差更小 |
| **`__syncthreads()`** | block 内的屏障，所有线程到达后才继续 | 共享数据协作的必要条件 |
| **warp shuffle** | warp 内线程直接交换寄存器值，不经过 shared memory | 比 shared memory 更快 |
| **atomicAdd** | 原子加，多个线程安全地更新同一位置 | 顺序不确定，导致非确定性 |
| **两阶段归约** | 各 block 先算部分和写入中间缓冲，再由第二个 kernel 合并 | 跨 block 归约的标准做法 |
| **grid-stride loop** | 一个线程按 grid 大小步进处理多个元素 | 让线程数独立于数据规模 |
| **Welford 算法** | 在线计算均值和方差的数值稳定算法 | LayerNorm 等统计算子的基础 |
| **并行合并** | 把两组 Welford 统计量合成一组：`M2 + delta²·nA·nB/n` | 多块统计合并的正确公式 |

## 算子

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **elementwise** | 逐元素运算，输出每个元素只依赖对应输入 | 最易并行，通常 memory bound |
| **broadcast** | 按规则把形状不同的张量对齐到同一形状 | 地址推进规则容易写错 |
| **type promotion** | 混合 dtype 时输出 dtype 的提升规则 | 影响精度与整除 |
| **softmax** | 归一化成概率分布；稳定实现需减最大值 | 数值稳定的经典案例 |
| **online softmax** | 单遍扫描即可算出 softmax，边扫边重缩放 | FlashAttention 的核心 |
| **LayerNorm / RMSNorm** | 沿指定维度归一化 | 统计合并是难点 |
| **GEMM** | 通用矩阵乘，`[M,K]@[K,N]` | 2MNK 次浮点运算；性能优化的主战场 |
| **tile / 分块** | 把大矩阵划成小块，提高数据复用 | GEMM 优化的第一手段 |
| **epilogue** | 矩阵乘之后的收尾（加 bias、激活） | 融合的常见位置 |
| **im2col** | 把卷积的输入展开成矩阵，复用 GEMM | 卷积与 GEMM 的桥梁 |
| **FlashAttention** | 分块计算注意力，不物化完整的分数矩阵 | 减少访存的核心思想 |
| **KV cache** | 缓存已算过的 K、V，避免 decode 时重算 | 自回归推理的关键结构 |
| **prefill / decode** | 处理整个前缀 / 逐 token 生成两个阶段 | shape 特性差异极大 |
| **causal mask** | 只允许看到当前位置及之前 | 按绝对位置比较，不是局部下标 |
| **GQA** | 多个 query 头共享一组 K/V 头 | 减少 KV cache 体积 |
| **SDPA** | PyTorch 的缩放点积注意力接口，按环境分派后端 | 名字里有 flash 不代表走了硬件实现 |
| **fusion** | 把多个算子合并成一个 kernel，避免中间结果落内存 | 减少搬运量 |

## 精度与量化

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **fp32 / fp16 / bf16** | 不同位宽的浮点格式 | 动态范围与精度的权衡 |
| **累积 dtype** | 累加器使用的精度，通常高于输入 | 属于契约，不是优化 |
| **mixed precision** | 计算用低精度、累加用高精度 | 常见工程做法 |
| **量化** | 把浮点映射到低比特整数 | 推理加速的主要手段 |
| **per-tensor / per-channel / per-group** | 量化参数的粒度 | 粒度越细精度越好、开销越大 |
| **scale / zero-point** | 量化映射的缩放与偏移 | 反量化的两个参数 |
| **PTQ / QAT** | 训练后量化 / 量化感知训练 | 两条量化路线 |
| **校准（calibration）** | 用代表性数据确定量化参数 | PTQ 的关键步骤 |
| **TF32** | 19 位浮点，用于 tensor core 的折中格式 | 默认开启时会影响数值 |

## 框架

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **schema** | 算子的签名定义（名称、参数、返回） | 注册的第一步 |
| **dispatcher** | 按 dtype/设备/布局把调用路由到具体实现 | 后端分派的机制 |
| **fake tensor / meta** | 只有形状和 dtype、没有数据的内存表示 | 用于形状推导和编译期检查 |
| **autograd** | 自动微分，需要注册反向公式 | 训练场景必需 |
| **torch.compile** | 图编译，需要算子可被追踪 | fullgraph 要求无图断裂 |
| **golden anchor** | 手算出的期望值常数 | 不依赖任何实现，能抓住两边一起错 |

## 平台与工具

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| **SASS / PTX** | 设备汇编 / 中间表示 | 确认编译器实际生成了什么 |
| **MMA** | 矩阵乘加指令，由 tensor core 执行 | 现代 GEMM 的性能来源 |
| **cp.async** | 异步拷贝，让访存与计算重叠 | 流水线的基础 |
| **TMA** | Hopper 起的张量内存加速器 | 更高维度的异步搬运 |
| **double buffering** | 双缓冲，一边算一边预取下一块 | 消除访存与计算的串行 |
| **persistent kernel** | 常驻 kernel，一次启动处理多批任务 | 摊薄启动开销 |
| **`__launch_bounds__`** | 告诉编译器每块线程数和最小块数 | 约束寄存器分配 |
| **ncu / nsys / compute-sanitizer** | 单 kernel 分析 / 时间线 / 正确性检查 | 三类不同问题用不同工具 |

## 相关页面

- [流程主线](../01-workflow/README.md)
- [基础域](../02-foundations/README.md)
- [决策树](decision-tree.md)：接到一个新算子该怎么做
- [常见错误清单](common-pitfalls.md)
