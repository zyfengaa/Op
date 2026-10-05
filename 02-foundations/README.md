# 基础域

不写代码也必须懂的概念。这一域是后面所有内容的地基。

## 页面

| 页面 | 回答 |
|---|---|
| [execution-model.md](execution-model.md) | SIMT、warp、block、grid、SM、occupancy、分支发散 |
| [memory-hierarchy.md](memory-hierarchy.md) | 内存层级、带宽、合并访存、bank conflict |
| [indexing-and-layout.md](indexing-and-layout.md) | shape、stride、地址推导、视图 |
| [quiz.md](quiz.md) | 自测 |

## 精度与语义契约

[contracts/](contracts/) 是[流程阶段一](../01-workflow/01-anatomy.md)的直接支撑：写算子契约时逐条对照。

| 页面 | 回答 |
|---|---|
| [contracts/dtype-and-precision.md](contracts/dtype-and-precision.md) | 浮点为什么不满足结合律、dtype 谱系、累加精度、容差怎么定 |
| [contracts/promotion-and-broadcast.md](contracts/promotion-and-broadcast.md) | 广播规则、类型提升、广播维度的地址推进 |
| [contracts/boundary-semantics.md](contracts/boundary-semantics.md) | 空输入、size-0、单元素、极值、NaN、非整除、对齐、超大 shape |
| [contracts/determinism.md](contracts/determinism.md) | 非确定性从哪来、怎么控制、什么时候可以放弃 |

## 按需要读

| 你遇到的情况 | 读 |
|---|---|
| 不理解「一个 block 里线程怎么编号」 | [execution-model](execution-model.md) |
| 有效带宽很低 | [memory-hierarchy](memory-hierarchy.md) |
| 地址算错、形状推错 | [indexing-and-layout](indexing-and-layout.md) |
| 结果和框架对不上 | [contracts/dtype-and-precision](contracts/dtype-and-precision.md) |
| 边界输入崩了 | [contracts/boundary-semantics](contracts/boundary-semantics.md) |
| 每次运行结果不一样 | [contracts/determinism](contracts/determinism.md) |

## 和其他域的关系

- **基础域**讲「为什么」——硬件和数据的行为。
- **[算子域](../03-operators/README.md)** 讲「怎么写」。
- **[性能域](../04-performance/README.md)** 讲「怎么量、怎么改」。

基础域的概念会在另外两域里反复出现；这一域只给一次完整解释，之后直接引用。

## 相关页面

- [术语表](../99-reference/glossary.md)
- [流程主线](../01-workflow/README.md)
- [常见错误清单](../99-reference/common-pitfalls.md)
