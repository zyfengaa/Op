# 01 · Hello World 算子

本阶段开始写 Kernel。建议选 CUDA 做第一个具体平台；学完一种执行/调试工具链后，再迁移 Triton 或国产平台。

## 课程

1. [Vector Add](01-vector-add/README.md)：内存分配、H2D/D2H、kernel launch、tail mask、reference 和计时。
2. [Elementwise Activation](02-elementwise-activation/README.md)：ReLU、Sigmoid、GELU、dtype 与融合。
3. [Reduction Sum/Max](03-reduction-sum-max/README.md)：第一个需要线程协作的算子。

## 本阶段交付物

第一次完整交付请照着 [Vector Add 五件套](01-vector-add/case-study.md) 和 [Reduce 五件套](03-reduction-sum-max/case-study.md) 做：每个案例连接实际版本源码、四个故障任务、原始 profiler 证据、第二后端和模型位置。

每个算子都要有公式、接口契约、reference、至少五种 shape、正确性输出、编译运行命令和性能基线。不能只提交一个 `.cu` 文件。

建议学习时间 2–4 周，依 C++ 和 GPU 环境基础调整。
