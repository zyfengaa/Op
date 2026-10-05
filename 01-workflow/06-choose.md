# 阶段六 · 选手段：从瓶颈到手法

## 这一步在防什么

跳过错这一阶段，就会变成“凭印象改代码”——看到一个常见优化就套上去，改完发现变慢了，也不知道为什么。

选手段的产物是一个**可证伪的预测**，不是一段代码。

## 预测的写法

一句话，三个部分，缺一不可：

> 瓶颈是 **X**，因此用手段 **Y**，预期指标 **Z** 会从 **A** 变到 **B**。

例子：

> 瓶颈是访存不合并（有效带宽 167 GB/s，峰值 900 GB/s），因此让同一 warp 访问连续地址，预期有效带宽会从 167 GB/s 升到 600 GB/s 以上。

**预测先写下来，再动手。** 事后补一个“我早就知道”的预测不算数。

**预测没中，说明对瓶颈的判断错了**——回阶段五重做定位，而不是改预测去迁就结果。

## 映射表

| 瓶颈（来自阶段五） | 候选手段 | 去哪查 |
|---|---|---|
| 访存不合并 | 调整访问顺序、向量化加载 | [techniques/memory-access](../04-performance/techniques/memory-access.md) |
| 数据复用低 | tile 分块、寄存器分块 | [techniques/occupancy](../04-performance/techniques/occupancy.md) |
| 访存与计算串行 | double buffer、`cp.async`、TMA | [techniques/pipelining](../04-performance/techniques/pipelining.md) |
| 寄存器压力限制 occupancy | 减少每线程工作量、`__launch_bounds__` | [techniques/occupancy](../04-performance/techniques/occupancy.md) |
| 启动开销占比高 | 算子融合、persistent kernel | [techniques/fusion](../04-performance/techniques/fusion.md) |
| 计算单元空置且 AI 高 | 提高复用、改用 MMA | [techniques/tensor-core](../04-performance/techniques/tensor-core.md) |
| 有效吞吐随规模非线性 | 先怀疑边界或同步 | 回 [阶段一](01-anatomy.md) |
| 已到设备上限 | 减少工作量，而不是换访问模式 | [阶段五](05-locate.md) 的“到顶”一列 |

最后两行是**否决项**。它们指向“这个方向没有收益”，能省下大量无效工作。

## 先估收益上限

动手前先问：**这个手段最多能带来多少提升？**

用 Amdahl 的思路：

~~~text
如果被优化部分占总时间的比例是 p，
提速比是 s，
整体最大提速 = 1 / ((1 - p) + p / s)
~~~

一个只占 8% 时间的阶段，哪怕优化到无限快，整体也只快 8.7%。如果整体目标是 2 倍，这个方向直接排除。

**先做大头。** 阶段四的 trace 已经告诉你各阶段占比，选手段时按占比排序，不要按“这个技巧很酷”排序。

## 一次只改一个变量

一次改动里包含多个手段，得到的结果无法归因——你不知道是哪个起作用，也不知道各自的贡献，更不知道其中一个是否在**拖后腿**。

本仓 GEMM 版本阶梯是反例的正写法：v1 → v2 → v3，每一版只引入一个机制（分块 → 向量化），所以每版的差异都能归因。

如果确实需要同时改多处（比如两个手段互相依赖），把它们当成**一个新的组合手段**，并且准备好单独验证失败时如何拆分。

## 什么时候不值得优化

- 已经到设备上限，且工作量无法减少。
- 该阶段占比很小，Amdahl 上限低于目标。
- 优化会让契约变模糊（比如改变累加精度的顺序）。
- 收益区间和噪声区间重叠——见 [阶段八](08-verify.md)。

**“不优化”是一个合法结论**，而且经常是正确结论。

## 常见误区

**不写预测就动手。** 事后无法判断是猜中的还是碰巧。

**同时改多个变量。** 见上。

**按技巧热度而不是按占比排序。** 先做占比大的。

**忽略 Amdahl 上限。** 花大力气优化一个占比 5% 的阶段。

**预测没中就改预测。** 那是自欺，不是分析。

## 去哪查

- [04-performance/techniques/](../04-performance/techniques/)：各手段的适用条件与代价
- [04-performance/counterexamples.md](../04-performance/counterexamples.md)：看起来该有用但实际没用的案例
- 下一阶段：[07-optimize.md](07-optimize.md)
