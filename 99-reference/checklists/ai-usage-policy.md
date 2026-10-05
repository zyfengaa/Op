# AI 辅助算子开发：使用规则与审查清单

## AI 适合承担的工作

- 生成样板、接口骨架和重复版本。
- 将清晰的数学公式翻译成初版 kernel。
- 补充测试组合和文档草稿。
- 解释编译错误、PTX/SASS 片段或 profiler 字段。
- 在明确目标语义下做 CUDA/Triton 代码对照。

## 人必须审查的部分

### Correctness

- shape、stride、layout 是否与真实输入一致？
- 尾块、空张量和小尺寸是否安全？
- shared memory 是否所有路径都初始化？
- barrier 是否所有 block 内线程一致到达？
- atomic 或并行 reduction 是否改变确定性/误差？
- dtype promotion、累加精度、NaN/Inf 行为是否符合契约？

### Performance

- 算法是 compute、memory、latency 还是 launch 受限？
- thread mapping 是否让访问合并？
- tile 与 shape 是否匹配？
- register/shared memory 占用是否影响并发？
- cache hit 是否有可利用的复用前提？
- pipeline 是否重叠了真正的等待？
- 优化是否降低绝对时间和端到端时间？

### Platform assumptions

- 设备架构、warp/wave 宽度、MMA 形状和异步拷贝要求是否匹配？
- API 是否适用于当前 toolkit/runtime？
- 生成代码是否真的走了目标矩阵指令？
- 编译产物和 profiler 是否确认了推断？

## 建议的 AI 工作流

1. 提供数学语义、输入约束、目标设备和可用工具版本。
2. 要 AI 明确列出它做出的硬件和精度假设。
3. 让 AI 先写最小 correctness 版本，不要求一开始就优化。
4. 人检查地址、mask、同步、dtype 和资源。
5. 执行 reference 对比和 sanitizer。
6. 保存 baseline，提出一条可测的性能假设。
7. 每轮只让 AI 修改一个主要变量。
8. profiler 验证假设；未成立就记录并回退。
9. 保存人对方案的修正与原因。

## 一个能亲自复现的审查练习

假设 AI 提供了三个建议，请先预测失败输入，再运行本仓库实验核对。

**建议一：GEMM 中输出越界的线程立刻 return。** 对普通无协作的逐元素 kernel 可能成立，但 tiled GEMM 的输出越界线程仍可能负责搬运别的线程需要的 A/B 值，并参与 block 屏障。用 M=N=K=3、TILE=2 画右下 tile，能发现合法 C[2,2] 需要其他线程搬来的数据。

**建议二：Online Attention 最大值改变时只缩放分母。** 用 score=[0,2]、V=[10,20]、每块一个 key，正确输出约 18.807971；漏缩放旧 o 会得到超出 V 范围的值。运行 dataflow_lab.py --topic attention，观察两个状态如何同时换基准。

**建议三：abs(actual-expected)>tol 就足以判断数值失败。** 对 actual=NaN，比较可能为 false，从而误判通过。先按契约检查是否允许非有限值，再使用容差；这不仅是 kernel 审查，也是测试代码审查。

这些案例说明，向 AI 提供完整契约还不够，验收者必须能设计能击中隐含假设的输入。提问日志应保留“为什么这个反例有效”，而不是只保存最终正确代码。

## 代码审查记录模板（填写）

~~~text
AI proposal:
隐含假设:
我检查了哪些索引/同步/dtype:
发现的问题:
修改内容及原因:
正确性证据:
性能证据:
是否接受及适用范围:
~~~

AI 是加速实现的协作者，不是硬件测量工具，也不能代替责任人签署 correctness 和 performance 结论。
