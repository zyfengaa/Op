# 性能实验室：把结论连到原始证据

本目录不是“应该能达到的性能”表，而是保存实际运行方法和已有采样。硬件、dtype、线程、输入驻留位置不同，成绩不同是正常现象；读报告时先检查实验条件。

## 实验一：一个算子，多个实现，跨尺寸反转

~~~powershell
python 09-practice/operator-stories/operator_story.py --output 09-practice/operator-stories/results/local
~~~

需要能导入 PyTorch 的 Python 环境，CPU 即可。脚本对 Add/Sum 比较 Python 循环和 PyTorch 原生 CPU；Sum 另有串行执行的 Python 树，Add 另有包含列表/tensor 转换的路径。它们先检查结果，再采 4 种长度、每种 6 个候选、每候选 9 个批平均样本。

打开 [已保存报告](results/cpu-reference/README.md)，先读 N=1 和 N=65536。练习回答：原生库为什么不是对每个尺寸都更快？库调用、对象创建与批量计算分别可能在哪种规模下更重要？你的判断中，哪些来自观察，哪些仍是假设？

## 实验二：局部改动，整体未必受益

~~~powershell
python 09-practice/model-slice/run.py --profile --output 09-practice/model-slice/results/local
~~~

进入 [模型实测解读](../model-slice/results/cpu-reference/README.md)，比较完整块的 reference 与候选，而不是只比较某一次 GEMM。现有数据的候选中位数略低，但样本范围大幅重叠；正确结论是证据不足以证明稳定收益，不是硬凑一个加速倍数。

## 产物的阅读顺序

1. summary.json 的 correctness：语义是否先对齐。
2. metadata 与 measurement：什么设备、源码、shape、线程、计时范围。
3. samples_us：是否只有一个好看的数字，或确实保留多次样本。
4. profiler.txt 与 trace.json：实际调用路径、父子事件和分配行为。
5. SVG 图：回到前四项解释图中差异；图不是独立证据。

trace 使用 Chrome trace 格式，可用兼容的 trace 查看器打开。先找 story/ 或 slice/ 用户范围，再展开原生事件；不要把所有嵌套 total 相加。报告可能包含本地路径或环境信息，分享自己的结果前先检查内容。

## “预期报告”不是一份固定百分比答案

可以预期 Add 的输出元素数、Sum 的标量输出、模型的阶段顺序和两份 KV 的逻辑形状。不能预设所有机器的占比、带宽或 occupancy 都应该相同。原始 trace 中存在某个事件，也不代表单凭名字就确定最终设备指令。

对 GPU 的报告仍需实际 GPU。当前 CPU 证据不能推导 HBM 带宽、bank conflict、寄存器数或 Tensor Core 利用率；这些检查回到 [GPU 测量章](../../04-performance/measurement.md)。

## 最常见的三个运行误区

- 找不到 torch：使用已经安装兼容 PyTorch 的 Python 环境；标准库故障题集仍可独立运行。
- 初始学生练习失败：它是待修复版本，先看 [练习入口](../fault_injection/README.md)，不是性能脚本故障。
- 测出的微秒数不同：先核对采样条件和系统负载，再讨论优化。不要同时运行多个 benchmark，也不要一边开 profiler 一边把耗时当成普通运行结果。

本机 PyTorch 会提示未安装 NumPy，但这些实验不调用 NumPy，已能通过实际检查；这不保证任何其他依赖警告都可以忽略。

## 交付给同学的复现包

保留运行命令、代码版本/哈希、原始样本、profile 和三条有限结论。明确哪些输入不支持、哪些设备未测试。若改了源码，需要重新采集，不能只更新解释文字使旧数据“支持”新版本。

继续阅读 [性能反例档案](../../04-performance/counterexamples.md)，或将自己的反转现象按相同方法加入新目录，不覆盖这里已有的基线。

## 参考资料

- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：输入准备、线程与计时方法。
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)：事件采集和 trace。
