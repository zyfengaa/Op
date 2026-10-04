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

## 6. 一个例子贯穿 schema、实现和 kernel

考虑 y=relu(x+bias)，x 为 [B,D]，bias 为 [D]。Python 调用者看到的是两个 Tensor 和一个 Tensor 输出；公共接口还应说明输入不被修改、输出是新 tensor、bias 按列广播。

schema 表达名称、参数和别名/修改约定。后端实现负责把语义落在 CPU、CUDA 或其他设备上。一个后端实现可能调用多个 kernel，也可能调用一个融合 kernel；因此“一个框架算子”与“一个 GPU kernel”没有固定的一一对应关系。反过来，框架编译器也可能把多个原始算子融合进一个设备 kernel。

把一段 torch 计算包装成自定义算子，首先得到的是新的接口边界，并不自动产生更快的机器码。性能结论仍要查看实际 launch 与数据流。

## 7. fake/meta 执行为什么不能读取真实数据

框架在图捕获或形状传播时，可能只有 shape/dtype/device/stride 等元信息，没有真实输入数值。此时仍要知道输出的元信息。fake 实现对本例返回同形、同 dtype、同 device 的新 tensor 即可；它不能通过读取 x[0].item() 来决定输出形状。

如果输出形状依赖数据内容，例如 nonzero 类操作，所需的框架集成比这个固定形状案例复杂。不要把 empty_like 的写法泛化到所有算子。

## 8. autograd 为什么不会凭空知道自定义边界里面的公式

对本例，设 z=x+bias，输出 relu(z)。当 z>0 时，dy/dz=1；z<0 时为 0；z=0 处需要遵守目标框架选定的约定。若上游梯度是 G，则：

~~~text
dX[b,d]    = G[b,d] * (X[b,d]+bias[d] > 0)
dBias[d]   = Σ_b dX[b,d]
~~~

bias 的梯度要沿 batch 维求和，是因为同一个 bias[d] 被 B 行共同使用。忘记这次归约会返回错误 shape 或错误值。保存输入可在 backward 重算激活 mask；保存 mask 则可能省去重算，但增加前向存储。训练集成因此也是存储与计算的权衡。

## 9. 本机可运行的框架实验

[custom_bias_relu.py](../examples/pytorch/custom_bias_relu.py) 注册了 Python 自定义算子、fake 实现和反向，并执行 forward 对照、gradcheck、opcheck、完整图捕获、非连续输入以及错误形状测试：

~~~powershell
python examples/pytorch/custom_bias_relu.py
~~~

它依赖 PyTorch，CPU 即可运行；本次开发环境为 PyTorch 2.8.0+cpu。它内部使用 PyTorch 运算，不是自行实现的融合 CUDA kernel。图捕获采用 eager backend，验证自定义边界能进入图，不用于证明编译加速。

本例输入 x=[[-2,1],[3,-4]]、bias=[0.25,0.5]，输出 [[0,1.5],[3.25,0]]。对 y.sum()，输入梯度为 [[0,1],[1,0]]，bias 梯度为 [1,1]。先在纸上求出这些结果，再阅读 backward 函数。

## 10. 为什么需要两种不同测试工具

gradcheck 通过数值差分检验梯度公式，宜使用 float64 和远离不可微点的输入。opcheck 检查算子注册与框架约定，例如声明的修改行为、fake 和 autograd 集成等。两者分别回答“导数是不是对的”和“框架能否按约定使用它”。

若 forward 对，但 fake 返回错误 shape，真实执行可能正常而图捕获失败。若注册没问题，但 dBias 漏了 batch 求和，opcheck 的某些检查可能通过而梯度数学仍不对。因此课程把两种结果分别记录，不用一个 PASS 代替另一个。

## 参考资料

- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- [PyTorch ATen source](https://github.com/pytorch/pytorch/tree/main/aten)
