# 练习层

可判定的练习。这里的共同点是：**有明确的通过标准**，不是读完打个勾。

练习与知识域**分开**是有意的——来查资料的人不该被练习题挡住，来练手的人需要一条完整的动手路径。

## 四个练习

| 练习 | 形式 | 需要 | 通过标准 |
|---|---|---|---|
| [故障注入](fault_injection/README.md) | 12 个故意写错的实现 | 标准 Python | 修改 `student.py`，让 39 组回归输入全部通过 |
| [代码阅读](reading/README.md) | 15 道判定题 | 标准 Python | 回答覆盖读写责任、同步、mask、越界 |
| [算子故事](operator-stories/README.md) | 实测 Python vs 原生库 | PyTorch | 解释跨尺寸反转，并保留原始样本 |
| [模型切片](model-slice/README.md) | 完整 Decoder Block | PyTorch | 7 项测试通过 + prefill/decode 对齐 |

## 建议顺序

```mermaid
flowchart LR
  A[故障注入<br/>建立"什么算修好了"] --> B[代码阅读<br/>训练读别人的 kernel]
  B --> C[算子五件套<br/>走完一个算子的交付]
  C --> D[模型切片<br/>把算子连成模型]
```

先做故障注入，因为它最快建立「可判定」的标准——你会看到判题脚本如何区分「修好了」和「看起来对了」。

## 快速开始

~~~powershell
# 只需要标准 Python
python 09-practice/check.py list
python 09-practice/check.py show VA01
python 09-practice/check.py grade VA01

# 需要 PyTorch
python 09-practice/operator-stories/operator_story.py --output 09-practice/operator-stories/results/local
python -m unittest discover -s 09-practice/model-slice -p "test_*.py" -v
python 09-practice/model-slice/run.py --profile --output 09-practice/model-slice/results/local
~~~

**初始判题失败是题目设计**，不是环境问题。`student.py` 是待修复版本。

## 已保存的实测数据

练习目录里保留了本机采样的真实数据，供对照：

| 数据 | 位置 |
|---|---|
| 算子故事：四尺寸 × 多实现的样本与图 | [operator-stories/results/cpu-reference](operator-stories/results/cpu-reference/README.md) |
| 模型切片：E2E、阶段 profile、trace、内存 | [model-slice/results/cpu-reference](model-slice/results/cpu-reference/README.md) |

这些是 **CPU 数据**，不能用来推断 HBM 带宽、bank conflict、寄存器数或 Tensor Core 利用率。见[能力矩阵](../00-start/environment.md)。

模型的候选实现中位数略低，但**样本区间大幅重叠**——正确结论是证据不足以证明稳定收益。这组数据保留下来，就是供练习如何拒绝过度解释。

## 练习与知识域的对应

| 练习 | 对应知识 |
|---|---|
| 故障注入 | [边界语义](../02-foundations/contracts/boundary-semantics.md)、[常见错误清单](../99-reference/common-pitfalls.md) |
| 代码阅读 | [执行模型](../02-foundations/execution-model.md)、[内存层级](../02-foundations/memory-hierarchy.md) |
| 算子故事 | [性能测量](../04-performance/measurement.md)、[性能反例](../04-performance/counterexamples.md) |
| 模型切片 | [03-operators](../03-operators/README.md)、[流程主线](../01-workflow/README.md) |

## 相关页面

- [流程主线](../01-workflow/README.md)：练习对应流程的哪一步
- [00-start](../00-start/README.md)：环境准备
- [checklists/stage-gates](../99-reference/checklists/stage-gates.md)：阶段验收
