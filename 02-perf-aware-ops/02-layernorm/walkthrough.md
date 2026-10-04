# LayerNorm 完整案例：两次归约与逐元素仿射

先用 [CPU 实验](../../examples/cpu/layernorm_lab.py) 跟随数字计算，再讨论 GPU 映射。学习目标是分清“均值/方差的统计归约”和“gamma/beta 的逐元素变换”，并能用常量行、非连续布局定位错误。

## 1. 输入契约与手算

一行 x=[1,2,3,4]，D=4。均值 μ=(1+2+3+4)/4=2.5。偏差为 [-1.5,-0.5,0.5,1.5]，平方为 [2.25,0.25,0.25,2.25]；总体方差 σ²=5/4=1.25。若暂设 eps=0，gamma 全为 1、beta 全为 0，则输出约 [-1.341641,-0.447214,0.447214,1.341641]。真实算子常设 eps>0，结果会略变；必须使用接口声明的 eps 才能精确比较。

本章采用总体方差，即除以 D，而不是样本方差的 D-1。D=1 时方差为 0，输出等于 beta（gamma 和输入有限时）；这也是非常有价值的边界测试。gamma/beta 的维度与最后一维 D 匹配，并对前面的行广播。

## 2. 为什么先两遍，再谈 Welford

直观的两遍算法先归约 x 求 μ，再归约 (x-μ)^2 求 σ²。公式 E[x²]-E[x]² 虽看起来可一遍完成，但当 x 都接近一个很大的数、彼此差值很小时，两个大数相减会丢失有效位。Welford 可逐步更新统计量，并可合并分块结果；它值得学习，但初学者应先有可验证的两遍 reference。

归一化公式是 (x-μ)/sqrt(σ²+eps)，然后乘 gamma、加 beta。把 eps 放在 sqrt 外面不是同一算子。方差为 0 的常量行有 eps 保护，归一化部分为 0；若 eps=0，不能简单宣称仍有效。

## 3. GPU 数据流

一行一个 block 的版本可先把一行加载到寄存器或共享存储，归约得到 μ，再计算局部平方并归约得到 σ²，最后每线程对负责的元素算 gamma/beta。行很长时，一个 block 的资源可能不足，必须考虑分段统计；行很短、行数很多时，每行一个 block 可能浪费线程。最后一块的 sum 填 0，同时除数仍是真实 D 而非补齐后的宽度。读取 gamma/beta 时也只读有效列。

所有阶段必须明确 dtype：FP16 输入可以在 FP32 中累加统计，随后按目标 dtype 写回。与 PyTorch 比较前，固定 eps、normalized_shape、layout 和仿射参数。训练场景还要实现并验证反向传播；本章 CPU 实验只覆盖前向。

## 4. 运行与测试设计

~~~powershell
python examples/cpu/layernorm_lab.py
python -m unittest discover -s examples/cpu
~~~

测试：D=1；常量行；[-1e4, 1e4]；非 2 的幂 D；多行且 gamma/beta 各列不同。对手算例子检查输出均值近 0、方差近 1（eps 非零时不会精确为 1）。仿射后的输出不再要求均值为 0 或方差为 1，这是常见误判。GPU 版还需在目标框架上测 transpose/slice 的 stride 语义。

## 5. 练习与答案要点

练习 A：x=[2,2,2]、gamma=[1,2,3]、beta=[4,5,6]，eps>0。输出 [4,5,6]。练习 B：x=[1,3]，μ=2，σ²=1；eps=0、gamma=[1,1]、beta=[0,0]，输出 [-1,1]。练习 C：为何补齐到 8 个 lane 的 D=5 不能除以 8？因为 padding 不属于输入，均值定义的除数是有效元素数 D。

## 参考资料（按学习顺序）

1. [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：核对方差、eps、normalized_shape 和 affine 语义。
2. [Triton LayerNorm Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)：学习行级映射和前后向。
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：并行归约与同步。
