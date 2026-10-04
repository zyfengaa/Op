# 学习阶段验收门槛

阶段时间仅作参考；达到能力门槛再进入下一阶段。

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
