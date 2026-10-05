# 阶段四 · 采集：拿到原始证据

## 这一步在防什么

阶段三给你一个可信的总数。它告诉你“多快”，但不告诉你“时间花在哪”。

采集是拿到细粒度证据的动作。它的产物是**原始数据**，不是结论——结论属于阶段五。

## 工具选择

三个工具回答三个不同的问题，不要混用。

| 工具 | 粒度 | 回答什么 | 什么时候用 |
|---|---|---|---|
| `nsys`（Nsight Systems） | 时间线 | kernel 之间的空隙、host↔device 行为、谁在等谁 | 怀疑有同步、拷贝或调度浪费 |
| `ncu`（Nsight Compute） | 单个 kernel | 吞吐、occupancy、访存效率、stall 原因 | 已经锁定是哪个 kernel，要找它内部的原因 |
| `compute-sanitizer` | 正确性 | 越界、竞争、同步错误 | 结果不对、或结果不确定地对 |

顺序通常是：先 `nsys` 看整体（哪个 kernel 占大头、有没有空隙），再 `ncu` 深入那一个。

**注意**：`compute-sanitizer` 属于正确性工具，不属于性能工具。它极慢，不要拿它的运行时间做性能判断。

## 采集纪律

**固定 workload。** 采集时的 shape、dtype、输入必须和阶段三测量时一致，否则两组数据对不上。

**profiler 与 benchmark 分开跑。** 一次采集只做一件事。

**不要同时跑多个采集。** 资源竞争会污染双方。

**保留原始输出。** 保存 profiler 表和 trace 文件，不只是截图或结论句。

## 读 trace 的三条规则

### 1. 区分子事件的 total 与 self

trace 里的父子事件，父的 total 包含所有子事件。**把所有嵌套事件的 total 相加会严重重复计数。**

要看“某个阶段实际占多少”，用 self time，或者只对**互不重叠**的兄弟事件求和。

### 2. 不要用 instrumented 占比当成硬件利用率

profiler 给出的是 CPU 侧的 instrumented 范围占所标注范围总和的比例。这个数**不是** GPU 的 SM 利用率，也不是 HBM 带宽占比。

### 3. 确认实际走了哪条路径

这一条最重要，也最容易被名字骗。

一个真实例子：模型切片里调用 `scaled_dot_product_attention`，看起来像是用了 FlashAttention。但原始 trace 里的实际事件是：

~~~text
aten::_scaled_dot_product_flash_attention_for_cpu
~~~

`for_cpu` 明确说明这是 CPU 路径，**不是** CUDA FlashAttention 的测量。如果只看外层函数名里有 `flash` 就下结论，会得出完全错误的判断。

做法：翻到原生事件层，看**实际执行**的那个名字，而不是你调用的那个 API 名字。

## 该保存什么

~~~text
原始 profiler 输出（按 kernel / 按阶段）
原始 trace（Chrome trace 格式即可）
采集时的环境与 shape
实际执行的原生事件名（用于确认路径）
~~~

本仓的报告在 `summary.json` 里保留了 `native_events` 字段，正是为了换设备或换版本时能重新确认分派结果。

## 常见误区

**先有结论再找证据。** 想验证“我优化有效”，于是只看支持它的那一个指标。

**只看一个指标。** 吞吐高了但 stall 更严重，可能是把问题挪了位置，不是解决了。

**把嵌套 total 相加。** 见上面规则 1。

**信 API 名字不信实际事件。** 见上面 `for_cpu` 的例子。

**把 profiler 开销当成被测对象的行为。**

## 去哪查

- [04-performance/tooling.md](../04-performance/tooling.md)：工具命令与读数口径
- [04-performance/toolchain-deep.md](../04-performance/toolchain-deep.md)：更深一层的 SASS 与编译选项
- 下一阶段：[05-locate.md](05-locate.md)
