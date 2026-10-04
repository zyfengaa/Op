# 08：框架集成：PyTorch、Triton 与算子契约

## 本章目标

把独立 CUDA kernel 变成框架可以调用、检查、回归和部署的算子。

## 集成边界

公共层负责：算子语义、shape/dtype/device 检查、CPU reference、误差标准、测试和 benchmark。后端层负责 CUDA kernel、编译器、runtime、同步和 profiler。这样迁移到 Triton、其他 GPU 或 NPU 时不需要重写测试契约。

## PyTorch 集成流程

1. 定义 Python API 和 schema。
2. 用 C++/CUDA extension 注册 operator。
3. 做 device、dtype、contiguous/stride 和 shape 检查。
4. 用 `torch.library.opcheck`、reference 对比和梯度测试验证。
5. 再做 micro benchmark 和真实模型 E2E 测试。

不要在 Python 包装层偷偷改变输入语义，也不要只测 contiguous tensor；真实模型可能产生非连续视图。

## Triton 学习路径

Triton 适合用较少的代码实验 block pointer、mask、program mapping、fused softmax 和 matmul。它降低了入口门槛，但不替代 shape 分析、正确性测试和 profiler。

## 参考资料

- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch Extension Overview](https://docs.pytorch.org/tutorials/extension.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
