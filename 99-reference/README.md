# 速查层

查得到，不用按顺序读。这一层是全库的索引设施。

## 页面

| 页面 | 用途 |
|---|---|
| [glossary.md](glossary.md) | **术语表**。按主题分组，每个词一句话定义 + 为什么重要 |
| [decision-tree.md](decision-tree.md) | **接到一个新算子该怎么做**。从「真的需要自己写吗」开始的决策树 |
| [common-pitfalls.md](common-pitfalls.md) | **常见错误清单**。按症状组织，含对应练习编号 |
| [reading-guide.md](reading-guide.md) | **论文与开源 kernel 源码导航**。从哪读起 |
| [external-links.md](external-links.md) | 官方文档与论文索引 |
| [checklists/](checklists/) | [阶段验收清单](checklists/stage-gates.md)、[AI 使用政策](checklists/ai-usage-policy.md) |
| [templates/](templates/) | [实验记录模板](templates/experiment-log-template.md)、[profiler 报告模板](templates/profiling-report-template.md)、[AI 提示记录模板](templates/ai-prompt-log-template.md) |

## 什么时候用

| 情况 | 去哪 |
|---|---|
| 看到一个不认识的词 | [glossary](glossary.md) |
| 拿到一个新算子任务 | [decision-tree](decision-tree.md) |
| 结果不对，想快速定位 | [common-pitfalls](common-pitfalls.md) |
| 想读原论文或抄一个成熟实现 | [reading-guide](reading-guide.md) |
| 要判断自己是否过关 | [checklists/stage-gates](checklists/stage-gates.md) |
| 要记录一次实验 | [templates/experiment-log-template](templates/experiment-log-template.md) |
| 要留证据给同学复现 | [templates/profiling-report-template](templates/profiling-report-template.md) |

## 和其他域的关系

速查层**不产生新知识**，只做索引和模板。所有内容都在对应的知识域里，这里只给最短的入口。

## 相关页面

- [流程主线](../01-workflow/README.md)
- [00-start](../00-start/README.md)：初次使用的入口
