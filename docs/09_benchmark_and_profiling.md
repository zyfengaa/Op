# 09：Benchmark、Nsight 与证据链

## 本章目标

建立可重复的性能实验，知道什么时候使用 Nsight Systems，什么时候使用 Nsight Compute，并能从指标反推出瓶颈。

## Benchmark 规则

- 固定 GPU、驱动、编译参数、频率策略和输入。
- 预热后再测，避免首次编译和缓存影响。
- 分开测 kernel、Host-device copy 和 E2E。
- 报告平均值、标准差或 P50/P90，不只报最好一次。
- 每个 shape 至少有 correctness 和性能两张表。

## 工具选择

Nsight Systems 用于回答“哪个阶段或哪个 Kernel 占总时间”；Nsight Compute 用于回答“这个 Kernel 为什么慢”。如果 Kernel 目标不清楚，先做全局时间线，再对 Top Kernel 做细粒度 profiling。

## 诊断顺序

```text
Top Kernel
→ Shape/dtype/layout
→ Arithmetic Intensity/Roofline
→ Compute/Memory/Latency/Launch 定性
→ bandwidth、MMA、occupancy、cache
→ global/shared memory access
→ register、bank conflict、stall
→ 修改一个变量并复测
```

不要从单一 utilization 指标下结论。性能报告至少要保存命令、设备信息、输入 shape、编译版本和 `.ncu-rep`/截图路径。

## 参考资料

- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
