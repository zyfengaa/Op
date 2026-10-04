# 学习阶段验收门槛

阶段时间仅作参考；达到能力门槛再进入下一阶段。

## 先看要留下什么证据

不以“读完章节”替代能力验收。每个阶段保留能复跑的代码、原始输入、检查结果和解释；设备不可用时将硬件项标为待验证，不填造性能数字。

| 实践关卡 | 自动检查 | 还需人工解释 |
|---|---|---|
| Add 完整案例 | VA01–VA04 学生修复通过 | ceil 覆盖、stride、广播轴、四元素尾部各解决什么 |
| Reduce 完整案例 | RD01–RD04 学生修复通过 | identity/count、奇数长度、两阶段全部成本 |
| 进阶故障 | SM01/LN01/GM01/AT01 通过 | 下溢、带权合并、协作装载、状态重缩放 |
| 陌生代码阅读 | 自己填写的答案达到 15/15 | 每题一句依据，不能只提交选项 |
| 性能证据 | 生成自己的 summary、trace 和图 | 分清实测/预测，解释一次性能反转或不明显收益 |
| 模型切片 | 7 项测试与 runner 对照通过 | prefill/decode shape、绝对位置、causal、cache 和计时范围 |

命令和题目见 [调试训练场](../exercises/README.md)、[模型主线](../06-model-slice/README.md)。运行参考解只验证题库，不能替代学生提交自己的实现。

## Gate 0 · 基础

- 能从公式写出 CPU reference。
- 能解释 shape、dtype、stride、误差。
- 能计算 row-major 地址和向量全局线程编号。
- 能区分 launch、传输、kernel 三类成本。

## Gate 1 · Hello World

- Vector Add 在长度 0/1/非 block 整数倍等输入正确。
- 能解释每个 thread 负责什么。
- 能报告 kernel-only 与端到端时间。
- 知道小输入为何可能受 launch bound 限制。

## Gate 2 · Reduce/Softmax

- 能解释 block 内同步范围。
- 能处理负数 max、尾部 padding 和空行语义。
- 能说明浮点误差与累加顺序。
- 能指出多阶段和 fused 实现的成本差异。

## Gate 3 · GEMM

- 能推导 M/N/K、FLOPs、输入输出字节数。
- naive kernel 对非整除 shape 正确。
- 能解释 tiling 为何带来复用，又消耗哪些资源。
- 有可复现 baseline 和一次有证据的优化记录。

## Gate 4 · 性能分析

- 能区分 Compute/Memory/Latency/Launch 受限的证据。
- 不从单个 utilization 指标直接下结论。
- 每次只改一个主要变量并保存报告。
- 能解释 kernel 加速与 E2E 加速为何不同。

## Gate 5 · 工程集成

- 接口契约完整，输入检查和错误信息合理。
- correctness、边界、dtype/layout 回归可执行。
- 区分 schema/dispatch 检查、数学正确性和梯度正确性。
- 新 backend 不需要复制所有公共测试。

## Gate 6 · AI 协作

- 能把 AI 生成代码中的关键索引、同步和 dtype 逐项解释。
- 能识别未经验证的硬件假设与过时 API。
- 留有假设、证据、修改和实测结果。
- 不把 AI 生成的 benchmark 数字当成设备实测。

## Gate 7 · 从算子到模型

- 能从 block 的数据流找到两次 RMSNorm、两次残差和全部投影。
- 能推导 QKV/MLP 的 M/K/N，以及 prefill 与单 token decode 的不同。
- 完整、分段和逐 token 执行在约定误差下对齐，且旧 KV Cache 不被修改。
- 能解释 SDPA 的实际分派证据，不把 API 名称直接当成 GPU 实现。
- 报告包含 E2E 的原始采样，阶段 profile 与 E2E 分开解读。
- 区分逻辑 KV 体积、allocator baseline/peak 和进程 RSS；没有测到的字段不冒充实测。

## 参考资料

- [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)：框架接入的测试层次。
- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：可复现测量与运行环境。
