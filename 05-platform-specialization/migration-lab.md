# 平台迁移实操：把 Vector Add 从公共语义带到新后端

本章不假定任何国产 GPU/NPU API 与 CUDA 完全等价。若尚未确定设备型号和 SDK 版本，就先完成公共层与测试；平台相关代码必须查目标设备的官方指南后再写。目标是让你知道迁移时“保留什么、重新证明什么”。

## 1. 冻结不随设备变化的契约

选择 Vector Add：输入 a、b 长度相同，输出 c[i]=a[i]+b[i]，允许 N=0 还是拒绝、支持哪些 dtype、是否支持非连续布局、输出是否可 alias 输入，都应先定下来。公共 CPU reference 与测试向量固定：N=0/1、线程组宽度附近、非整除长度、极值。设备不能改变数学契约，但不同精度或计算顺序可能导致容差不同；应在规格中明确允许范围。

## 2. 分层目录的建议

~~~text
contract/          语义、shape/dtype/layout、错误条件
reference/         CPU 计算
tests/             跨后端共用的输入和验收
backends/cuda/     CUDA 编译、stream、kernel、profiling
backends/target/   目标 SDK 编译、queue、kernel、profiling
benchmarks/        同一工作负载的端到端测量
~~~

本仓库还没有实际目标设备 adapter，这个布局是教学设计，不是声称已实现。先把公共 reference 跑通，再逐个后端复用同一套测试。

## 3. 逐项核对平台差异

不要照搬 blockDim、warpSize=32 或 __syncthreads 的字面概念。逐项查：线程/任务组如何定义；内存空间如何寻址；数据搬运是否显式、异步及其同步规则；向量/矩阵指令支持的 dtype 与 tile；host runtime 如何分配、launch、等待和报错；编译器的代码生成、调试器、profiler；设备内存容量与带宽。若目标文档未证实某项能力，就标为待验证，不从 CUDA 经验直接推断。

把原 CUDA 的“每元素独立 + ceil 网格 + 越界 mask”翻译为目标平台的工作分配；若目标平台采用不同任务划分，可以改变映射，但仍要让每个有效输出恰好写一次，且无效 lane 不读写越界。

## 4. 迁移的验收顺序

第一步只要求 target 后端在 N=1 和非整除 N 上与 reference 一致。第二步加更多 dtype/layout/极值。第三步用目标工具检查越界、竞争和同步。第四步在该设备上测 warmup 后的 kernel 与端到端时间，记录 SDK、驱动、编译 flags。第五步才探索 tile、向量化、异步搬运。不要拿两台不同设备的绝对延迟直接证明“某语言更好”；需控制工作负载并讨论各自硬件和软件栈。

若换成 Reduce，额外检查目标线程组的归约原语、同步粒度和全局结果合并；若换成 GEMM，额外检查矩阵指令布局、累加类型和尾块。每升一级，都复用公共测试，但追加平台特有边界。

## 5. 练习和答案要点

练习 A：一个教程写死“warp=32，所以每组只要做五轮二分归约”，迁移时能否直接用？不能；目标组宽和同步原语必须查官方文档与实际编译结果。练习 B：target 的 Vector Add 比 CUDA 慢，能否立刻归因于硬件？不能；先确认输入传输、launch、测量范围、频率、编译配置和设备带宽。练习 C：CPU reference 与后端输出相同，是否证明无数据竞争？不能；还需设备侧检查和并发压力测试。

## 参考资料

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：作为已学平台的对照。
2. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)：观察另一种编程抽象。
3. 目标设备的官方编程、运行时、编译、profiling 文档：实施前记录具体设备型号、SDK 版本和文档 URL；未定平台时不能提供准确的 vendor API 指南。
