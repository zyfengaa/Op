# 参考资料索引

优先阅读官方指南、论文原文和高质量参考实现。链接会随软件版本变化；涉及具体 API、指标名称和硬件约束时，应以当前 toolkit/设备版本文档为准。

## 通用 GPU 与 CUDA

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：执行模型、内存、同步、异步、Tensor Core。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存、性能指标与优化建议。
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)：可编译示例。
- [CUTLASS](https://github.com/NVIDIA/cutlass)：高性能 GEMM/卷积参考实现。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：内存、竞争和同步检查。

## Triton

- [Tutorial gallery](https://triton-lang.org/main/getting-started/tutorials/)：Vector Add、Softmax、Matmul、LayerNorm、Attention。
- 阅读时记录 Triton 版本和 GPU 型号；API 和最佳配置会演进。

## Profiling

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)：全局时间线、CPU/GPU 排队、kernel 选择。
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)：单 kernel 指标。
- [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)：从症状逐层定位，不从单一指标下结论。

## Framework

- [PyTorch custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：接口注册、opcheck 和集成。
- [PyTorch custom Python operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)：Python 层 schema/fake/autograd 相关能力。
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)：教学型 kernel 实现。

## 进阶论文

- [FlashAttention](https://arxiv.org/abs/2205.14135)：IO-aware attention。
- 阅读论文必须区分算法本身、实验硬件、baseline 和测量口径；不能直接把论文速度数字当作本仓结果。

## 国产平台

对 Ascend C、SDAA、BANG C 等平台，仓库暂不提供未经核实的统一 API 结论。进入平台专精阶段后，应由目标硬件和 SDK 版本驱动，链接对应厂商官方开发指南、编译器文档、profiling 文档和 sample。
