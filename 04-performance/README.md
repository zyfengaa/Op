# 性能域

跨算子通用的性能知识。算子域回答“这个算子怎么写”，性能域回答“为什么慢、怎么改”。

## 四个部分

| 页面 | 回答 | 何时读 |
|---|---|---|
| [measurement.md](measurement.md) | 怎么测出可信数字 | 建立基线、做验收 |
| [analysis.md](analysis.md) | 从数字到瓶颈类型 | 拿到 profiler 数据后 |
| [tooling.md](tooling.md) | 用哪个工具、怎么读 | 需要采集证据时 |
| [toolchain-deep.md](toolchain-deep.md) | 编译器与 SASS 层 | 需要确认实际生成了什么 |
| [techniques/](techniques/) | 具体优化手段与其成立条件 | 已定位瓶颈，要动手 |
| [counterexamples.md](counterexamples.md) | 看起来该有用但实际没用的案例 | 动手前先避坑 |

## 与其他域的分工

- **[流程主线](../01-workflow/README.md)** 讲“什么时候做这一步”。
- **性能域** 讲“这一步具体怎么做”。
- **[算子域](../03-operators/README.md)** 讲“这个算子的特有瓶颈”。

三者的关系：流程是时间轴，性能域是方法库，算子域是案例库。

## 核心链条

~~~text
测量 → 采集 → 定位 → 选手段 → 实施 → 验收
  ↑                                      |
  └──────────── 迭代 ────────────────────┘
~~~

每一环在[流程主线](../01-workflow/README.md)里有对应的阶段页。

## 三条最容易违反的纪律

**1. 不同计时范围的数字不可比。**

kernel-only、op-level、E2E 三者包含的内容不同。混用会让差异里混进范围差。见 [measurement.md](measurement.md)。

**2. occupancy 不是目标。**

它是隐藏延迟的手段。牺牲数据复用来换更高 occupancy，时间可能反而变长。

**3. 区间重叠时结论是“证据不足”。**

不是“略有提升”。见[流程阶段八](../01-workflow/08-verify.md)。

## 手法索引

| 瓶颈 | 手法 |
|---|---|
| 访存不合并 | [memory-access](techniques/memory-access.md) |
| 数据复用低 | [occupancy](techniques/occupancy.md) |
| 访存与计算串行 | [pipelining](techniques/pipelining.md) |
| 启动开销占比高 | [fusion](techniques/fusion.md) |
| 计算单元空置且 AI 高 | [tensor-core](techniques/tensor-core.md) |
| 需要减少搬运总量 | [fusion](techniques/fusion.md) |
| 生产者/消费者职责混在一起 | [warp-specialization](techniques/warp-specialization.md) |
| 小 batch 反复启动、权重可常驻 | [persistent-kernels](techniques/persistent-kernels.md) |

## 本仓的实测数据

- [性能反例档案](counterexamples.md)：跨尺寸反转的真实样本
- [算子故事实测](../09-practice/operator-stories/README.md)：Python vs 原生 CPU
- [模型切片实测](../09-practice/model-slice/README.md)：阶段 profile 与 E2E

**注意**：这些是 CPU 数据，**不能**用来推断 HBM 带宽、bank conflict、寄存器数或 Tensor Core 利用率。见[能力矩阵](../00-start/environment.md)。

## 参考资料

- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)：指标与 stall 原因口径
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存与 occupancy
- [Roofline 模型](https://dl.acm.org/doi/10.1145/1498765.1498785)：算术强度与上界
