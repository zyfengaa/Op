# 06 · 平台专精：迁移共同方法，重学平台机制

先完成通用路线，再根据明确硬件选择专精方向。

先做 [已运行的第二后端实验](second-backend.md)：在 Python reference 和 PyTorch 原生 CPU 上执行同一份 add/sum 契约，观察转换成本和性能反转。再读 [平台迁移实操](migration.md)，把这一经历推广到设备映射。原生 CPU 实验不等于已验证国产 NPU 或某种 SIMD 指令。

## 共同能力

跨平台可迁移：数学语义、shape/stride 契约、CPU reference、误差验证、并行分解、tiling 思路、Roofline 推理、benchmark 设计和假设验证。

## 平台专属内容

需要按目标平台重新学习：线程组模型、地址空间、同步语义、compiler、runtime、矩阵指令、异步数据移动、profiling 和高性能库。

CUDA/Triton 可作为起点，因为公开教学材料丰富；它们不是其他平台的规范。昇腾 Ascend C、太初 SDAA、寒武纪 BANG C 等应以用户实际 SDK 和厂商当前文档为准。

## 迁移练习

选一个已完成的 Vector Add 或 Reduce：

1. 冻结数学契约和测试向量。
2. 新建 backend adapter，不改公共 reference。
3. 实现设备版并复用公共 correctness cases。
4. 分别记录编译/runtime/profiler 差异。
5. 不搬用 CUDA 的 warp、bank、MMA 固定假设。
6. 对相同输入重新测量，不比较未经控制的不同机器峰值。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- 目标硬件厂商对应版本的官方编程、runtime 和 profiling 指南；开始平台专精时填入具体链接和 SDK 版本。
