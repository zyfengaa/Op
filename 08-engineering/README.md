# 工程化：从 kernel 到可维护算子

算子写完只是开始。这个域回答的是：**怎么让别人能信任、复现和接手你的实现**。

先读 [完整算子交付案例](end-to-end-project.md)：以 LayerNorm 为例串起接口、reference、边界测试、性能记录与提交审查。CI 相关见 [CI 与可复现交付](ci.md)。

## 工程目标

把算子语义、CPU reference、测试契约和 benchmark 与后端实现分开。CUDA 是一个 backend；Triton 或其他加速卡可以增加新 backend，而不复制所有测试和接口。

## 四条边界

| 边界 | 容易混淆的地方 | 卡在哪 |
|---|---|---|
| 语义 vs 实现 | 契约写在注释里而不是接口上 | 换后端时契约丢失 |
| CPU reference vs 设备实现 | 用设备结果反过来定义 reference | 错误被一致地复现两次 |
| 测试 vs benchmark | 用 benchmark 当回归测试 | CI 因噪声抖动而红 |
| 本仓数据 vs 你的环境 | 把 CPU 微秒数当成设备性能 | 结论不可迁移 |

## 本仓已经提供的入口

| 入口 | 位置 | 需要 |
|---|---|---|
| CUDA 构建与 CTest | [`assets/cuda/`](../assets/cuda/CMakeLists.txt) | CUDA Toolkit |
| CPU 语义测试 | [`assets/cpu/`](../assets/cpu/) | 标准 Python |
| 故障注入题库 | [09-practice/fault_injection](../09-practice/fault_injection/README.md) | 标准 Python |
| 框架接入实验 | [custom_bias_relu.py](../05-frameworks/impl/custom_bias_relu.py) | PyTorch |

框架侧（注册、fake、autograd、opcheck、图捕获）的原理见 [05-frameworks](../05-frameworks/README.md)，本文档不重复。

## 相关页面

- [流程主线](../01-workflow/README.md)：交付对应流程的哪一步
- [性能测量](../04-performance/measurement.md)：性能记录要带哪些元数据
- [AI 使用政策](../99-reference/checklists/ai-usage-policy.md)：哪些环节不能交给模型
- [阶段验收](../99-reference/checklists/stage-gates.md)：每个阶段的合格判据

## 参考资料

- [PyTorch custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
