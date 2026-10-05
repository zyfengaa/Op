# 性能分析：Roofline 与瓶颈分类

这一页回答：**拿到一组 profiler 数字之后，怎么判断瓶颈属于哪一类、下一步往哪个方向走**。

前置条件：

- 有一个可信的实测时间，且计时范围已写明（[measurement.md](measurement.md)、[阶段三](../01-workflow/03-measure.md)）。
- 有一份原始采集，能看到吞吐、occupancy、访存计数与 stall 分布（[tooling.md](tooling.md)、[阶段四](../01-workflow/04-profile.md)）。

分析分三步，缺一步结论都是猜的：

~~~text
第一步：算上界      → 这个负载的理论上限在哪
第二步：用实测反推   → 离上界多远（利用率）
第三步：看 stall     → 谁在拖后腿，裁决候选瓶颈
~~~

## 1. 三个定义

~~~text
算术强度 AI   = FLOPs / 搬运字节数          （FLOPs per byte）
机器平衡点     = 峰值算力 / 峰值带宽         （FLOPs per byte）
Roofline 上界  = min(峰值算力, AI × 峰值带宽)
~~~

| 量 | 含义 | 谁决定 |
|---|---|---|
| AI | 每搬运一个字节能做多少次浮点运算 | 算法与数据布局 |
| 机器平衡点 | AI 到达多少时，带宽线与计算线同时到顶 | 设备（峰值算力 ÷ 峰值带宽）|
| Roofline 上界 | 该负载在这台设备上的理论天花板 | 两者共同决定 |

| AI 与平衡点 | 上界由谁给出 | 瓶颈归属 |
|---|---|---|
| AI < 平衡点 | AI × 峰值带宽（带宽线） | memory bound |
| AI > 平衡点 | 峰值算力（计算线） | compute bound |
| AI = 平衡点 | 两条线交点 | 同时到顶 |

**搬运字节数用算法本身需要的量**（读多少、写多少），不是 profiler 计数器的实际值。后者的差异正是第二步要看的对象。

### 1.1 峰值必须取同一条路径

FP32 标量、FP16/BF16 矩阵（tensor core）、FP8、稀疏计算，各有完全不同的峰值。用某条路径的峰值去评判另一条路径的 kernel，会得出「利用率只有 5%」这类无意义的结论。

| 被测路径 | 峰值该取哪一条 | 常见错误 |
|---|---|---|
| FP32 标量运算 | FP32 CUDA core 峰值 | 拿 FP16 tensor core 峰值当分母 |
| FP16 / BF16 矩阵 | 对应 dtype 的 tensor core 峰值 | 拿 FP32 峰值当分母 |
| FP8 / INT8 矩阵 | 对应低精度 tensor core 峰值 | 忽略缩放与累加器开销 |
| 稀疏 | 稀疏路径峰值 | 与稠密峰值混用 |

**峰值来自设备规格书。** 本仓当前环境无 GPU（见[能力矩阵](../00-start/environment.md)），实测列只能标为「待实测」，不得用 CPU 数字或宣传峰值填充。

## 2. 用实测时间反推有效吞吐

~~~text
有效带宽 = 实际搬运字节数 / 实测时间
有效算力 = 实际 FLOPs     / 实测时间
利用率   = 有效值 / 同路径峰值
~~~

**实际搬运字节数取 profiler 的访存计数器**（例如 DRAM 读字节 + 写字节），并注明是哪一层（DRAM / L2 / L1）。同一 kernel 在 DRAM 层和 L2 层的「带宽」可以差好几倍，两者不可比。

一个手算例子（数字为演算，不代表具体设备）：

~~~text
负载：C = A + B，N = 2^24，fp32
FLOPs    = N            = 16.78 M
搬运字节 = 读 4N + 读 4N + 写 4N = 12N = 201.3 MB
设实测时间 = 300 us
有效带宽 = 201.3 MB / 300 us ≈ 671 GB/s
设峰值带宽 = 900 GB/s → 利用率 ≈ 75%
AI = 16.78 M / 201.3 MB ≈ 0.083 FLOPs/byte（= 1/12）
~~~

AI 只有 1/12，远低于任何现代 GPU 的平衡点，所以 elementwise 算子必然是 memory bound。这里利用率 75% 已经不错，但还没到顶。

## 3. 判据表

下表把「观察」映射到「瓶颈类型」和「下一步方向」。**「是否到顶」这一列是最重要的区分**：它决定继续优化是否有意义。

| 观察 | 判据 | 瓶颈类型 | 是否到设备上限 | 下一步方向 |
|---|---|---|---|---|
| 有效带宽远低于峰值 | 低于峰值 60%，且 AI < 平衡点 | Bandwidth Bound，**访存低效** | 否 | 查访问模式与合并度（[memory-access](techniques/memory-access.md)）|
| 有效带宽接近峰值 | 达到峰值 80%–90% | Bandwidth Bound，**已到顶** | 是 | 只能减少搬运量：融合、降精度、改算法 |
| 有效算力远低于峰值 | 低于峰值 50%，AI > 平衡点，计算单元大量空置 | Compute Bound（发射 / 依赖） | 否 | 查 ILP、依赖链、指令混合 |
| 有效算力接近峰值 | 达到峰值 80% 以上 | Compute Bound，**已到顶** | 是 | 只能减少计算量或换更快的单元 |
| 两者都远低于峰值，occupancy 低 | 并行度不足以隐藏延迟 | **Latency Bound** | 否（延迟尚未到顶）| 提高并行度、改用流水线（[occupancy](techniques/occupancy.md)、[pipelining](techniques/pipelining.md)）|
| 两者都远低于峰值，stall 以 `barrier` 为主 | 同步开销或负载不均占比高 | 同步 / 调度问题 | — | 减少屏障、改块内分工 |
| 该 kernel 各项利用率都低，但时间占比很小 | 它不是大头 | 非瓶颈 | — | 回到 nsys 找真正的热点 |

表中的 60% / 80% 是本页采用的**判读门槛**，是经验约定，不是硬件定律。它们只用来分档，最终裁决仍要看 stall（第 4 节）。门槛与设备、dtype、kernel 大小都有关，报告里应写明你用的是哪一档。

**「已到顶」为什么单列。** 到达设备上限时，在同一个方向继续优化是徒劳的：带宽已经 90%，再调访存模式几乎没有空间；算力已经 90%，再调指令调度也几乎没有空间。此时唯一有效的是**减少工作量**——融合以减少搬运、降低 dtype、换算法复杂度更低的实现。把「到顶」误判成「低效」，会让人反复去做无效的微调。

## 4. warp stall 原因表

occupancy 不是目标，它只是隐藏延迟的手段。真正裁决瓶颈类型的是 warp stall 分布。下表口径取自 Nsight Compute 的 Warp State Statistics（名称随版本略有差异）。

| stall 原因 | 含义 | 指向 | 下一步 |
|---|---|---|---|
| `long scoreboard` | 等全局 / 局部访存返回（长延迟） | 访存**延迟**或带宽 | 合并、向量化、预取；或提高并行度以掩盖延迟 |
| `barrier` | 等 `__syncthreads()` 或其它屏障 | 同步开销、块内负载不均 | 减少屏障、改分工、用 warp 级原语替代 |
| `MIO throttle` | MIO 指令队列满：shared memory 与特殊函数（如 MUFU）端口饱和 | 片上资源争用 | 消除 bank conflict、减少 shared 访问、改用 shuffle |
| `not selected` | warp 已就绪，但发射器选了别的 warp | 就绪 warp 过多或发射带宽受限 | 常与其他 stall 合看；若它独占且慢，考虑降低块内 warp 数 |
| `wait` | 等前一条定长延迟指令的结果（依赖链） | ILP 不足 | 增加独立指令、打断依赖链、展开循环 |

读这张表的规则：

- 看**占比最高**的那一个，不是看平均值。
- 单一 stall 占比明显过半时有清晰指向；多个并列，问题可能是复合的。
- `long scoreboard` 高**不等于**带宽低——它指向延迟。带宽高不高要看有效带宽（第 2 节）。
- `not selected` 孤立看意义有限：它只说明「有别的 warp 被选中」，可能是并行度充足的正常现象，也可能是过度订阅。

## 5. occupancy 不是目标

一个真实的反直觉例子：

~~~text
Kernel A：occupancy 75%，Tensor Core 利用率 55%
Kernel B：occupancy 50%，Tensor Core 利用率 85%
→ B 更快
~~~

B 用更低的 occupancy 换来了更高的每线程数据复用和指令级并行。**目标是「有足够的 warp 在飞行中以隐藏延迟，并让计算单元持续有活干」，不是「occupancy 越高越好」。**

| 证据 | 方向 |
|---|---|
| occupancy 低 + stall 以 `wait` / `not selected` 为主 | 提高并行度有收益 |
| occupancy 低 + stall 以 `long scoreboard` 为主 | 提高并行度可能仍有收益（更多 warp 掩盖延迟）|
| occupancy 低 + stall 以 `barrier` 为主 | 问题是负载不均或同步太多，不是占用率 |
| occupancy 高但仍然慢 | **不要再堆 occupancy**，先看数据复用和到顶与否 |
| 有 register spill | 优先消除 spill，它比 occupancy 更直接 |

## 6. 三个走完的算例

三个算例都用同一套三步法，且都区分「观察」与「假设」。

### 算例一：访存低效

`row-softmax`，shape `[4096, 1024]`，fp32。

~~~text
理论搬运 = 读 4096×1024×4 B + 写 4096×1024×4 B = 16 MiB + 16 MiB = 32 MiB
实测时间 = 200 us
有效带宽 = 32 MiB / 200 us ≈ 167 GB/s
峰值带宽 = 900 GB/s → 利用率 18.6%
~~~

**观察**：有效带宽只有峰值 18.6%，远低于 60% 门槛；AI 为个位数，低于平衡点。
**判读**：Bandwidth Bound，且**访存低效**（不是到顶）。
**假设**：同一 warp 访问的地址不连续——行跨度 4 KiB，若每个线程取一行里相隔较远的列，一个 warp 会触达多个内存段。
**下一步**：带着假设进 [memory-access](techniques/memory-access.md)；用 profiler 的访存事务计数验证（请求事务数远大于理想值）。若假设成立，改线程到地址的映射后有效带宽应显著上升。

### 算例二：已经到顶

同形状，优化访问模式后：

~~~text
有效带宽 = 810 GB/s，峰值 900 GB/s → 利用率 90%
~~~

**观察**：有效带宽落进 80%–90% 的上限带。
**判读**：Bandwidth Bound，**已到设备上限**。
**下一步**：再调访存模式、再加向量化，收益都极小。要更快只能**减少搬运量**——例如与上游算子融合，让中间结果不出内存（见 [fusion](techniques/fusion.md)）；或降低中间结果的 dtype。

**对比算例一与二**：同样是低 AI、同样是 Bandwidth Bound，一个能改访问模式，一个只能改搬运量。分岔点就是判据表的「是否到顶」列。

### 算例三：看起来是计算，其实是延迟

一个 GEMM，AI 算下来高于平衡点，本该是 Compute Bound。但实测有效算力只有峰值的 15%。

~~~text
观察：有效算力 = 峰值 15%（远低于 50% 门槛）
      occupancy = 12%
      stall 首位 = long scoreboard
错误假设：算力不够 → 去优化计算、换指令
修正：AI 高于平衡点只说明「上界由计算线给出」，
      不说明「已经到达计算线」。15% 意味着离上界很远，
      而 occupancy 12% + long scoreboard 说明
      没有足够的 warp 去掩盖访存延迟。
结论：Latency Bound，不是 Compute Bound。
方向：提高并行度（occupancy、pipelining），而不是优化计算。
~~~

**这就是为什么第三步不能省。** 前两步给出候选类型，stall 数据给出裁决。

## 7. 常见误区

**把 occupancy 当目标。** 它是手段。见第 5 节的反例。

**只算 AI 不看利用率。** AI 只告诉你上界在哪，不告诉你离上界多远。AI 高于平衡点的 kernel 也可能因为并行度不足只跑到 15%。

**把「低效」当成「到顶」。** 看到 80% 带宽就说没救了，可能其实还能到 90%。

**把「到顶」当成「低效」。** 反过来浪费时间在无效微调上。

**忽略 stall 数据直接下结论。** 见算例三。

**用错路径的峰值。** 拿 FP16 tensor core 峰值去评判一个 FP32 标量 kernel，得到的利用率没有意义。

**分母用理论字节还是计数器字节不声明。** 两种「带宽」数值不同、含义不同，混在一起不可比。

**在 CPU 数据上套 GPU 峰值。** CPU 实测数字不能用来推断 HBM 带宽、occupancy 或 Tensor Core 利用率，见[能力矩阵](../00-start/environment.md)。

## 相关页面

- [measurement.md](measurement.md)：计时范围与有效带宽的定义
- [tooling.md](tooling.md)：用哪个工具、怎么取数
- [toolchain-deep.md](toolchain-deep.md)：从 SASS 确认编译器实际生成了什么
- [techniques/memory-access.md](techniques/memory-access.md)、[techniques/occupancy.md](techniques/occupancy.md)、[techniques/pipelining.md](techniques/pipelining.md)：定位之后的具体手段
- [counterexamples.md](counterexamples.md)：看起来该有用但实际没用的案例
- [../01-workflow/05-locate.md](../01-workflow/05-locate.md)：这一步在流程里的位置
- [../99-reference/glossary.md](../99-reference/glossary.md)：AI、Roofline、stall 等术语
