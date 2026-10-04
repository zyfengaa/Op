# Kernel Fusion：什么时候值得融合

先读 [Fusion 逐步实验](walkthrough.md)，以 bias+ReLU 为例检查结果、字节数和测量设计。

## 融合解决什么问题

对 A→B 两个逐元素阶段，分开的实现可能写出中间 tensor，再由第二个 kernel 读回。融合可减少 launch 和中间结果流量：

~~~text
separate: read X → write T → read T → write Y
fused:    read X → compute A then B → write Y
~~~

## 不是所有融合都更快

融合可能增加寄存器压力、降低 occupancy、扩大代码体积，或让不同 shape 无法复用同一实现。决策应比较完整路径的时间、traffic 和资源。

## 分析步骤

1. 在时间线上确认这些 kernel 确实占可观成本。
2. 计算中间 tensor 字节数和额外 launch 数。
3. 确认数学变换不改变 dtype、舍入、NaN 或 alias 语义。
4. 实现 fused 与 unfused 版本。
5. 比较总时间、traffic、寄存器和 occupancy。
6. 对短小与大型 shape 分别决定是否 dispatch 到融合版本。

## 练习

融合 bias+ReLU、bias+GELU 或 GEMM+epilogue。记录中间流量、launch 数、数值误差和 E2E 耗时。

## 参考资料

- [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
