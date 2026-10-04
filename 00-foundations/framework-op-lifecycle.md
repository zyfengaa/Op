# 算子在深度学习框架中的生命周期

## 1. 从 Python 表达到硬件执行

一次常见的框架调用可概括为：

```text
Python API
→ dispatcher 根据 device/layout/dtype 选择实现
→ 参数和 schema 检查
→ CPU/CUDA/其他后端实现
→ runtime 将工作排入 stream/queue
→ kernel 执行
→ 输出 tensor 返回给后续算子
```

具体路径随框架版本、编译选项、图捕获和后端而变。这里是理解边界的简图，不代表所有调用都经过完全相同的内部函数。

## 2. 算子契约要写什么

对一个自定义算子，至少写清：

- 输入/输出 shape 和广播规则。
- 支持的数据类型与累加类型。
- contiguous 要求，或输入 stride 的处理方式。
- 输入输出 device 是否必须相同。
- 是否原地修改输入，是否允许 alias。
- 数值误差和 NaN/Inf 语义。
- 是否支持 autograd、torch.compile、vmap 等框架功能。

## 3. Forward 与 backward

训练算子可能需要定义反向梯度；推理专用算子则通常不需要 autograd。不要把“前向正确”误认为“训练集成完整”。PyTorch 的 opcheck 主要检查自定义算子注册和使用约定，不替代梯度数学正确性测试；有梯度时还需要 gradcheck 或独立梯度 reference。

## 4. 为什么先学习接口契约

很多故障不是算术错误，而是输入实际 non-contiguous、dtype 与 kernel 假设不符、输出别名错误或 stream 同步不正确。先定义契约，才能写出有针对性的检查和测试。

## 5. 小练习

给 `y = relu(x)` 写一个算子契约：shape、dtype、layout、device、in-place 行为、NaN 语义和测试列表。

## 参考资料

- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- [PyTorch ATen source](https://github.com/pytorch/pytorch/tree/main/aten)
