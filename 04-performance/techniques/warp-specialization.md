# 手法 · warp 分工

**适用瓶颈**：同一 kernel 内既有搬运又有计算，两者争抢同一批 warp 的发射槽——[阶段五](../../01-workflow/05-locate.md)判定 stall 中 `long scoreboard` 与 `not selected` / `MIO throttle` 同时出现，且加了流水后收益不达预期。

## 症状

前三条是**观察**，后两条是待验证的**假设**：

- 观察：每个 warp 内既发 load/`cp.async` 指令、又做计算，时间线上同一 warp 在等待与计算之间交替。
- 观察：`MIO throttle`（或 LSU 端口饱和）与计算端口空置**同时**出现。
- 观察：已经做了双缓冲/多级流水，但重叠率仍上不去，或者需要越来越复杂的屏障才勉强工作。
- 假设：把搬运集中在少数 warp、计算集中在其余 warp 后，两类发射能真正并行。
- 假设：搬运字节数足以用少数 warp 覆盖（先算字节比，再看 profiler 确认）。

**判断该不该分工**：若搬运的发射槽占用与计算的发射槽占用在时间上高度重叠（都接近饱和），分工才有意义。若搬运只占少数发射周期，分工带来的复杂度不值得。

## 手段

### 1. producer / consumer 分工

**producer warp** 只负责把数据从全局搬到 shared memory（发 `cp.async` 或 TMA）；**consumer warp** 只从 shared 读数并计算。两者走不同的代码路径。

**成立条件**：

- 搬运总量足够大，少数 producer warp 就能覆盖，且不成为新瓶颈。
- block 线程数足以拆出两个组，且不浪费（例如拆完后 consumer 的 occupancy 仍能隐藏计算延迟）。
- 有明确的多级 buffer 让 producer 领先 consumer（至少 2 级）。

### 2. 与流水线配合

分工几乎总与 `cp.async` / TMA 一起用：producer 持续填充多级 buffer，consumer 按序消费。

**成立条件**：

- 已经实现（或同时实现）双缓冲或多级流水，见 [pipelining](pipelining.md)。
- buffer 深度決定 producer 能领先的块数；深度不足时 producer 会追上 consumer，退化成串行。
- 没有流水时单独做分工，收益很小——因为分工要解决的正是「同一 warp 交替等待」，没有流水就没有可重叠的对象。

### 3. 同步方式改为命名屏障 / mbarrier

用 arrive/wait 对（`bar.arrive` / `bar.sync`、`cuda::barrier`、TMA 配套的 `mbarrier`）替换全块 `__syncthreads()`。

**成立条件**：

- 只需要**部分线程组之间**同步。全块 `__syncthreads()` 会强制 producer 停下来等 consumer，把分工的收益抹掉。
- arrive 与 wait 的计数严格配对（生产者到达数、消费者等待数）。
- 有明确的超时/终止路径，避免死锁。

### 4. 角色配比按字节比估算

**成立条件**：用「搬运字节 : 计算字节」的比值先估一个 producer/consumer 比例，再用 profiler 的实际时间占比校正。**具体配比必须实测，不能照抄别的 kernel 的数字**——它取决于 tile 大小、dtype 和指令吞吐。

## 代价

| 代价 | 机制 | 后果 |
|---|---|---|
| 可编程性下降 | 角色分支、buffer 状态机增多 | 维护成本上升，回归风险变大 |
| occupancy 计算变复杂 | 不同角色 warp 的寄存器/shared 需求不同，寄存器按所有角色的**最大值**分配 | 实际 occupancy 可能低于预期 |
| 调试变难 | 死锁来自 arrive/wait 计数不配对 | 需要专门排查屏障逻辑 |
| 负载不均时角色空转 | producer 处理完但 consumer 未算完，或反之 | 一种单元闲置 |

## 什么时候不要做

- **负载简单、单 warp 就能覆盖的小 kernel。** 分工只增加复杂度，没有可拆的独立工作量。
- **搬运占比很低。** 先算字节比；搬运在总发射周期中占比小的时候，分工收益被角色开销吞掉。
- **还不熟悉 mbarrier / 命名屏障。** 先做可靠的双缓冲版本，确认瓶颈确实在角色争抢，再加分工。
- **没有多级 buffer 基础。** 分工与流水是配套手段，单独上分工通常无效。
- **无法为死锁提供终止路径。** 屏障配对错误会让 kernel 挂死，且难以定位。

## 验证方式

| 指标 | 期望变化 |
|---|---|
| kernel 时间 | 下降（最终判据） |
| consumer warp 的 `long scoreboard` | 下降（搬运被隔离到 producer） |
| 搬运发射与计算发射的重叠 | 上升 |
| `barrier` stall | 下降（若用命名屏障替代全块同步） |
| 死锁 / 数据竞争 | 无（用 compute-sanitizer 的 racecheck、synccheck） |
| 每线程寄存器数与 achieved occupancy | 记录变化（可能下降；只要时间下降即方向正确） |

**若时间没降**，检查角色配比是否失衡，或 producer 是否成了新瓶颈。不要通过继续加深 buffer 掩盖配比问题。

## 常见误区

**分工了却仍用 `__syncthreads()` 同步。** producer 被 consumer 拖住，重叠消失。

**角色配比凭感觉。** 应先用 profiler 看搬运与计算各自的发射/时间占比，再定比例。

**忽略寄存器按所有角色的最大值分配。** 以为 consumer 寄存器少就能拿到高 occupancy，实际被 producer 路径拉高。

**arrive 与 wait 计数不配对，或某分支提前退出。** 直接死锁或读到未就绪数据。

**把 warp specialization 当默认最优。** 简单 kernel 上它只是增加复杂度，收益为零。

## 相关页面

- [性能域索引](../README.md)：手法与瓶颈的映射表
- [pipelining](pipelining.md)：分工常与多级流水、`cp.async`/TMA 一起用
- [occupancy](occupancy.md)：不同角色寄存器需求如何影响驻留块数
- [瓶颈分析](../analysis.md)：stall 组合如何指向角色争抢
- [阶段五 · 定位](../../01-workflow/05-locate.md)：`long scoreboard` 与 `MIO throttle` 的读法
- [counterexamples](../counterexamples.md)：加了机制却没变快的案例
- [术语表 · cp.async / TMA](../../99-reference/glossary.md)
