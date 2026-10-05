# 手法 · 流水线与双缓冲

**适用瓶颈**：访存与计算串行，硬件单元交替空转——[阶段五](../../01-workflow/05-locate.md)判定为 Latency Bound，或 Bandwidth Bound 但 stall 以 `long scoreboard` 为主、时间线上搬运段与计算段不重叠。

## 症状

前三条是**观察**（时间线或 profiler 直接给出），后两条是待验证的**假设**（先估算，再采集证据）：

- 观察：时间线呈「搬运—计算—搬运」交替，同一时刻只有一类单元在工作。
- 观察：stall 以 `long scoreboard` 为主（等全局访存返回）。
- 观察：有效带宽与有效算力都低于峰值，但两者各自占用时间的**和**接近 kernel 总时间。
- 假设：把某次全局加载换成常量后 kernel 明显变快，说明搬运未被隐藏。
- 假设：单块 tile 的搬运时间与计算时间同量级（比值在 0.5–2 之间），否则流水收益有限。

**先估收益上限**：串行时 `T = T_load + T_compute`，完全重叠后 `T = max(T_load, T_compute)`。因此上限提速比为：

~~~text
上限提速比 = (T_load + T_compute) / max(T_load, T_compute)

搬运与计算等量      → 上限 2.0 倍
搬运是计算的 1/4    → 上限 1.25 倍
~~~

第二条说明：**搬运占比低于 20% 时，流水最多快 25%，通常不值得动手。** 这是动手前的否决依据，不是实测结论。

## 手段

### 1. 双缓冲（double buffering）

用两份（或多份）shared memory tile，算第 k 块的同时预取第 k+1 块。

**成立条件**：

- 单块 tile 的搬运时间与计算时间同量级（上面的假设成立）。
- shared memory 能放下**至少两份** A tile 和两份 B tile。
- 块内搬运与消费之间有明确的屏障边界（见手段 3）。

**原理**：缓冲从「一份、用完才能再填」变成「两份、交替」，填一块与用另一块在时间上重叠，把 `T_load + T_compute` 压向 `max(T_load, T_compute)`。

### 2. 异步搬运：`cp.async` 与 TMA

用异步拷贝替换同步 `LDG`，让「发出搬运」与「等待完成」分离。

`cp.async`：把数据从全局直接搬入 shared memory，绕过寄存器，且**发出拷贝的 warp 不被阻塞**。

~~~cuda
// 双缓冲：commit 一组拷贝，计算期间去 wait 上一组
cp.async.cg.shared.global [smem_buf0], [gmem + k0], 16;
cp.async.commit_group();
// ... 用 buf1 计算第 k-1 块的收尾 ...
cp.async.wait_group(0);   // 只等还未完成的组
__syncthreads();          // 组内数据对所有线程可见
~~~

TMA（Hopper 起）：`cp.async.bulk` / TMA 用一个线程发起整个多维 tile 的搬运，其余 warp 完全不必参与发指令，完成通过 mbarrier 通知。

**成立条件**：

- `cp.async`：设备计算能力 sm_80 及以上；目标是 **shared memory**，不是寄存器（普通 `LDG` 仍是同步的）；搬运地址连续且 16 字节对齐，否则退化成普通加载，还多花一次屏障。
- TMA：设备 sm_90 及以上；tile 是多维或足够大（小 tile 的固定开销占比高）；已使用 mbarrier 做完成同步，不能再用单纯的 `__syncthreads()` 兜底。

### 3. 屏障位置随流水改变

单缓冲时，屏障在「上一块用完」处，一道就够。双缓冲/多级流水后，屏障要**拆成两类**：

| 屏障 | 保护的对象 | 何时用 |
|---|---|---|
| 消费者屏障 | 缓冲已被消费，可被下一轮搬运覆盖 | 搬运者等待它才覆写 |
| 生产者屏障 | 数据已就位，可被消费 | 消费者等待它才读取 |

用 `cp.async.wait_group`（或 mbarrier 的 arrive/wait）表达「这一组拷贝完成」，再用 `__syncthreads()` 让组内可见性对所有线程生效。**只加 `cp.async` 而不改屏障位置**，等于没有流水。

### 4. 流水级数：2 级 vs 3 级

| 流水级数 | shared 占用（每份 tile 为 S 字节） | 可隐藏的搬运延迟 | 何时用 |
|---|---|---|---|
| 1（无流水） | 1×S | 0 | 搬运占比低于 20%，不该做流水 |
| 2 | 2×S | ≤ 1 个 `T_compute` | 搬运延迟 ≤ 单块计算时间 |
| 3 | 3×S | ≤ 2 个 `T_compute` | 搬运延迟 > 单块计算时间，2 级填不满 |

**权衡**：级数与 shared 占用成正比。若 SM 的 shared 容量为 C，则每 SM 可驻留 block 数 ≤ `C / (级数 × S)`。**加深流水会等比压低 occupancy**（见 [occupancy](occupancy.md)）。当 occupancy 已不足以隐藏其它类型的延迟（如 `barrier`、指令依赖）时，加级数的收益可能被完全抵消。

## 什么时候不要做

- **计算已远长于访存**（`T_load / T_compute < 1/4`）。此时上限提速不足 25%，先按 Amdahl 排除方向，见 [06-choose](../../01-workflow/06-choose.md)。
- **访存本身不合并**。流水隐藏的是**延迟**，不修复**带宽**。访问模式低效时先做 [memory-access](memory-access.md)。
- **shared memory 不足**。强行加深流水把 occupancy 压到无法隐藏其它延迟，反而更慢。
- **没有明确的屏障边界**。无法安全地表达「缓冲已就位 / 已可覆写」，流水只会引入竞争。
- **负载已靠多 block 自然重叠**。两种运算分布在不同 block 时，硬件已经在隐式流水，再显式做一遍没有增量。

## 验证方式

| 指标 | 期望变化 |
|---|---|
| kernel 时间 | 下降（最终判据） |
| 时间线上搬运与计算的重叠率 | 上升 |
| stall `long scoreboard` | 下降 |
| 有效带宽与有效算力 | 同时上升 |
| 每 block shared memory 用量 | 上升（级数的代价，需记录） |
| achieved occupancy | 可能下降；只要时间下降即方向正确 |
| 尾块/非整除 shape 的正确性 | 全部通过 |

**若重叠率上升但 kernel 时间没降**，说明瓶颈不在这里，回[阶段五](../../01-workflow/05-locate.md)重新定位，而不是继续加级数。

## 常见误区

**加了 `cp.async` 却没改屏障位置。** 数据未就绪就被消费，或靠全块 `__syncthreads()` 兜底，把异步拷贝重新变回同步，流水失效。

**把「时间线有重叠」当成收益。** 重叠只是机制，最终判据是 kernel 时间。

**级数越多越好。** shared 占用与级数成正比，occupancy 被等比压低。

**对非连续地址用 `cp.async`。** 退化成普通加载，还多付一层屏障成本。

**只优化搬运占比很小的 kernel。** 先算上限提速比，低于目标直接排除。

## 相关页面

- [性能域索引](../README.md)：手法与瓶颈的映射表
- [瓶颈分析](../analysis.md)：Roofline 与瓶颈分类
- [occupancy](occupancy.md)：流水级数与 shared 占用如何影响驻留块数
- [memory-access](memory-access.md)：先修带宽，再隐藏延迟
- [counterexamples](../counterexamples.md)：看起来该快但没快的案例
- [阶段六 · 选手段](../../01-workflow/06-choose.md)：什么时候值得做这个方向
- [术语表 · double buffering / cp.async / TMA](../../99-reference/glossary.md)
