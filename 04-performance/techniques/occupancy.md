# 手法 · occupancy 与寄存器压力

**适用瓶颈**：Latency Bound——有效带宽和有效算力都远低于峰值，且 occupancy 低、stall 以等待为主。

## 先纠正一个前提

**occupancy 不是目标。** 它是隐藏延迟的手段。目标是“有足够的 warp 在飞行中，让计算单元不空转”。

一个真实的反例：

~~~text
Kernel A：occupancy 75%，Tensor Core 利用率 55%
Kernel B：occupancy 50%，Tensor Core 利用率 85%
→ B 更快
~~~

B 用更低的 occupancy 换来了更好的数据复用和指令级并行。**牺牲复用去堆 occupancy，是常见的方向性错误。**

## 两个方向

### 提高 occupancy（当并行度确实不足时）

occupancy 由每 SM 能驻留的 block/warp 数决定，受三个资源限制，取最小值：

| 资源 | 限制 |
|---|---|
| 寄存器 | 每线程寄存器数 × 线程数 ≤ 寄存器文件 |
| shared memory | 每 block 用量 × block 数 ≤ 片上容量 |
| 线程/block 上限 | 硬件规定的每 SM 最大线程数 |

**判断是谁在限制**：查 profiler 的 occupancy 部分，它会给出“limited by registers / shared memory / block size”。

### 降低寄存器压力

寄存器是 occupancy 最常见的限制来源。

**手段**：

- 减少每线程的工作量（比如从每线程 8 个 accumulator 降到 4 个）
- 用 `__launch_bounds__(threads, minBlocksPerSM)` 让编译器为更高的 block 数优化
- 缩短变量生命周期，让编译器能复用寄存器

**代价**：每线程工作量降低 → 数据复用下降 → 可能反而更慢。**这就是上面反例的机制。**

**`__launch_bounds__` 的用法**：

~~~text
__launch_bounds__(256, 4)   // 每块 256 线程，每 SM 至少 4 块
→ 编译器会把寄存器数压到 (寄存器文件 / (256 × 4)) 以内
→ 超过就 spill 到 local memory
~~~

**注意 spill**：寄存器不够时编译器会把变量放到 local memory（实际在全局内存上），这会静默地大幅拖慢。查 profiler 的 register spill 计数。

## 什么时候该做

| 证据 | 方向 |
|---|---|
| occupancy 低 + stall 是 `wait`/`not selected` | 提高并行度有收益 |
| occupancy 低 + stall 是 `long scoreboard` | 提高并行度可能仍有收益（更多 warp 掩盖延迟） |
| occupancy 低 + stall 是 `barrier` | 问题是负载不均或同步太多，不是占用率 |
| occupancy 高但慢 | **不要**再堆 occupancy |
| 有 register spill | 优先消除 spill |

## 验证方式

| 指标 | 期望 |
|---|---|
| achieved occupancy | 上升 |
| stall `wait` / `not selected` | 下降 |
| register spill | 为 0 |
| kernel 时间 | **下降**——这才是最终判据 |

**如果 occupancy 上升但时间没降，说明方向错了**，回到[阶段五](../../01-workflow/05-locate.md)重新定位。

## 常见误区

**把 occupancy 当成 KPI。** 见上。

**不看是谁在限制就直接加线程。** 如果是 shared memory 限制，加线程无用。

**不看 spill 就减寄存器。** 减到 spill 反而更慢。

**忘了权衡复用。** 每线程工作量降下去，数据复用也降下去。

## 相关页面

- [memory-access](memory-access.md)：访存本身的效率
- [execution-model](../../02-foundations/execution-model.md)：SM、warp、延迟隐藏
- [pipelining](pipelining.md)：用异步搬运代替堆并行度
