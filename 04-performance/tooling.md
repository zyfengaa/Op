# 工具链与采集：nsys、ncu、compute-sanitizer

这一页讲**用哪个工具、怎么采集、怎么读**。它对应[流程阶段四](../01-workflow/04-profile.md)的执行细节；把采集结果翻译成瓶颈类型，是[阶段五](../01-workflow/05-locate.md)和 [analysis.md](analysis.md)的事。

## 1. 三工具分工

三个工具回答三个不同的问题，**不要混用，也不要拿错工具的输出去支撑错类型的结论**。

| 工具 | 粒度 | 回答什么 | 产物 | 代价 |
|---|---|---|---|---|
| `nsys`（Nsight Systems）| 整个程序的时间线 | kernel 之间的空隙、host↔device 行为、谁在等谁、拷贝与同步 | `.nsys-rep`、sqlite / Chrome trace | 轻，可常态采集 |
| `ncu`（Nsight Compute）| 单个 kernel | 吞吐、occupancy、访存效率、stall 原因 | `.ncu-rep` 报告 | 重，多次 replay |
| `compute-sanitizer` | 单次运行的正确性 | 越界、竞争、同步错误 | 诊断文本 | **极慢，比正常慢几个数量级** |

**`compute-sanitizer` 是正确性工具，不是性能工具。** 它检测的是「有没有错」，不是「快不快」。它通过插桩和串行化来放大竞态与越界，运行时间与正常执行差几个数量级。**绝不能用它的运行时间去比较两个实现的性能**，也不要用它来解释某处的耗时。用它得到结论「没错」，不代表「快」。

## 2. 典型顺序

~~~text
nsys  →  找到时间大头和空隙，锁定一个候选 kernel
  ↓
ncu   →  深入那一个 kernel：吞吐 / occupancy / 访存 / stall
  ↓
analysis.md  →  把数字翻成瓶颈类型，决定下一步方向
~~~

先 nsys 再 ncu，是因为 ncu 很贵，而且只能看一个 kernel。先用便宜的工具确定「该看哪一个」，再把贵的算力花在它身上。

**如果 kernel 之间有空隙、或 host 在等 device、或大量时间花在 memcpy / 同步上，那属于 nsys 的领地。** 此时 ncu 报告再漂亮也没用——问题不在核里，在核外。这正是「单 kernel 快、整体慢」的典型成因。

## 3. 常用 ncu 指标分组

`ncu` 把指标组织成 section。下面几组是最常用的，覆盖从「到没到顶」到「为什么慢」。

| section | 回答 | 关键读数 |
|---|---|---|
| SpeedOfLight（SOL）| 计算与内存各用了峰值的多少 | SM throughput %、Memory throughput % |
| Occupancy | 实际驻留 warp 与上限之比 | Achieved Occupancy、限制来源（registers / shared / block size）|
| Memory Workload Analysis | 访存去了哪、效率如何 | 各层级流量、L1 / L2 / DRAM 命中率、sector 利用率 |
| Scheduler Statistics | 每周期发射几条指令、够不够 | Issued Warp Per Scheduler、Eligible Warps Per Scheduler |
| Warp State Statistics | warp 为什么停下来 | 各 stall reason 的占比（对照 [analysis.md](analysis.md) 第 4 节）|

**SOL 的两个百分比要看清单位。** SM throughput 是计算单元相对其峰值的占比，Memory throughput 是内存子系统相对峰值的占比。两者的分母不同，不能直接比大小；它们只回答「各自离自己的顶多远」。

常用命令形式：

~~~bash
# 先看本机有什么，不要照抄别的版本的 section / metric 名
ncu --version
ncu --list-sections
ncu --list-metrics

# 完整采集一个 kernel：输出很长
ncu --set full -k regex:row_softmax --launch-count 1 -o softmax-profile ./build/app

# 只要几组：更快、更聚焦（推荐）
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    -k regex:gemm ./build/app

# 直接点名指标
ncu --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/app
~~~

**`--set full` 的输出会很长，而且往往需要多次 replay（每个 replay 重置并重跑 kernel），不要全量照单全收。** 按需取段：先看 SOL 判断方向，再按方向取 Occupancy / Memory / WarpState。`--launch-count` 限制采集次数，避免对预热循环和全部 correctness 形状反复采集。

**metric 与 section 名随版本和设备变化。** 遇到 `unsupported metric`，先用 `--list-metrics` 查本机可用项，而不是把别的架构报告里的名字抄过来。

`nsys` 侧的常用形式：

~~~bash
nsys profile -o trace ./build/app           # 采集时间线
nsys stats trace.nsys-rep                    # 统计表
~~~

`compute-sanitizer` 侧的常用形式（本仓 [measurement.md](measurement.md)已有可运行版本）：

~~~bash
compute-sanitizer --tool memcheck   ./build/app   # 越界、非法访问
compute-sanitizer --tool racecheck  ./build/app   # shared memory 竞争
compute-sanitizer --tool synccheck  ./build/app   # 同步错误
~~~

## 4. 采集纪律

**固定 workload。** 采集时的 shape、dtype、输入与运行模式，必须和[阶段三](../01-workflow/03-measure.md)测量时完全一致。否则两边的数对不上，差异里混进了 workload 差。

**profiler 与 benchmark 分开跑。** 一次只做一件事。profiler 插桩会改变运行时间，插桩后的耗时**不是**被测对象的真实耗时（本仓模型报告里插桩后的单次 RoPE 达到 1130.1 us，比不带 profiler 的整块中位数还高，这个数字说明的是插桩开销）。

**不要并行多个采集。** 多个 profiler 或多个 benchmark 同时跑会互相争抢资源，双方数字都失真。

**保留原始输出文件。** 保存 `.nsys-rep`、`.ncu-rep`、sqlite / trace 和采集时的环境与 shape，而不是只留一句结论。事后换设备或换版本时，原始文件能让你重新确认分派结果。

## 5. 读 trace 的三条规则

### 规则一：父子事件的 total 包含子事件，不能把所有嵌套 total 相加

trace 里父子事件的 `total` 是**包含**子事件的累计时间。把整棵树里所有事件的 total 相加，会严重重复计数。

~~~text
model_forward  total = 100 ms
├── layer0      total =  20 ms   （layer0 自己的 self = 20 ms，无子事件）
└── layer1      total =  30 ms
把所有 total 相加 = 100 + 20 + 30 = 150 ms  ← 超过真实 E2E，错了
~~~

正确做法二选一：

- 要「某节点自己占多少」：用 **self time**（total 减去直接子事件的时间）。
- 要「一组阶段各占多少」：只对**互不重叠的兄弟事件**求和。上例中取 layer0 与 layer1 的 total 相加是合法的，因为它们不互相包含；把 model_forward 也算进去就重复了。

### 规则二：instrumented 占比不是 GPU SM 利用率

profiler 给出的 instrumented 比例，是**被插桩计时的调用范围之和**占所标注范围的比例。它发生在 CPU 侧，随插桩粒度和代码路径而变。

这个数**不是** GPU 的 SM 利用率，也不是 HBM 带宽占比，也不是「这段代码占 GPU 时间的比例」。要看 GPU SM 利用率，去 `ncu` 的 SpeedOfLight，或 `nsys` 的 GPU 时间线。

### 规则三：确认实际走了哪条路径

这一条最重要，也最容易被名字骗。

一个真实例子：模型切片里调用 `scaled_dot_product_attention`，看名字像是用了 FlashAttention。但原始 trace 里实际执行的事件是：

~~~text
aten::_scaled_dot_product_flash_attention_for_cpu
~~~

`for_cpu` 明确说明这是 **CPU 路径**，不是 CUDA FlashAttention 的测量。如果只看外层函数名里有 `flash` 就下结论，会得出「FlashAttention 在这台设备上跑了某某时间」这种完全错误的判断。

做法：翻到**原生事件层**，看**实际执行**的那个名字，而不是你调用的那个 API 名字。本仓的报告在 `summary.json` 里保留了 `native_events` 字段，正是为了换设备或换版本时能重新确认分派结果。

## 6. 常见误区

**拿 `compute-sanitizer` 的运行时间比较性能。** 它是正确性工具，插桩极慢。见第 1 节。

**先有结论再找证据。** 想证明「我的优化有效」，于是只挑支持它的那一个指标。

**只看一个指标。** 吞吐高了但 stall 更严重，可能是把问题挪了位置，不是解决了。

**把嵌套 total 相加。** 见规则一。

**把 instrumented 占比当成硬件利用率。** 见规则二。

**信 API 名字不信实际事件。** 见规则三的 `for_cpu`。

**把 profiler 采集出的时间当普通运行时间。** 插桩和 replay 都有开销，诊断用的数字不能进性能结论。

**用 `--set full` 的出结论，不看 section 名对应的口径。** 不同 section 的分母不同，混着解读会串味。

## 相关页面

- [measurement.md](measurement.md)：计时范围、warmup、带宽定义
- [analysis.md](analysis.md)：把采集数字翻成瓶颈类型
- [toolchain-deep.md](toolchain-deep.md)：从 SASS 层确认编译器实际生成了什么
- [../01-workflow/04-profile.md](../01-workflow/04-profile.md)：采集在流程里的位置
- [../01-workflow/05-locate.md](../01-workflow/05-locate.md)：读完 trace 之后怎么定位
- [../00-start/environment.md](../00-start/environment.md)：无 GPU 时这些工具不可用
