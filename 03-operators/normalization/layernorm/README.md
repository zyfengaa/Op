# LayerNorm：组合 Reduction 与 Elementwise

先读 [LayerNorm 完整案例](walkthrough.md)：两遍统计的手算、数值稳定、GPU 映射、测试和答案。

## 定义和契约

对最后一维长度 D 的每行：

~~~text
mean = sum(x) / D
var = sum((x - mean)^2) / D
y = (x - mean) / sqrt(var + eps) * gamma + beta
~~~

接口要声明方差定义、epsilon 的位置、gamma/beta 广播维度、累加精度、输出 dtype 和支持的 stride。

## 实现路线

1. CPU reference，使用清晰的两遍计算。
2. GPU 多阶段：求均值、求方差、归一化。
3. Fused 行 kernel：尽量把中间值留在寄存器或 shared memory。
4. 长行和短行分别调度；一个 CTA 一行不是对所有 D 都合适。

Welford 算法可稳定地并行合并均值与方差，但要完整实现其合并公式，不可随意替换成 E[x²]-E[x]² 并假设数值稳定。

## 验证

- 与 PyTorch LayerNorm 对比。
- 测 D=1、非对齐 D、大 D、常量行和大动态范围输入。
- 分别检查统计量误差与最终输出误差。
- 训练用途另测梯度；推理用途明确说明 autograd 支持范围。

## 常见错误

- 方差约定与框架不一致。
- epsilon 加在错误位置。
- 低精度累加造成误差明显放大。
- 对非连续 tensor 假设 stride 等于宽度。

## 参考资料

- [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Triton tutorials: Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/)
