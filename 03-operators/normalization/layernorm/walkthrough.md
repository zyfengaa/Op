# LayerNorm 完整案例：两次归约与逐元素仿射

先用 [CPU 实验](../../../assets/cpu/layernorm_lab.py) 跟随数字计算，再讨论 GPU 映射。学习目标是分清“均值/方差的统计归约”和“gamma/beta 的逐元素变换”，并能用常量行、非连续布局定位错误。

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
python assets/cpu/layernorm_lab.py
python -m unittest discover -s assets/cpu
~~~

测试：D=1；常量行；[-1e4, 1e4]；非 2 的幂 D；多行且 gamma/beta 各列不同。对手算例子检查输出均值近 0、方差近 1（eps 非零时不会精确为 1）。仿射后的输出不再要求均值为 0 或方差为 1，这是常见误判。GPU 版还需在目标框架上测 transpose/slice 的 stride 语义。

## 5. 练习与答案要点

练习 A：x=[2,2,2]、gamma=[1,2,3]、beta=[4,5,6]，eps>0。输出 [4,5,6]。练习 B：x=[1,3]，μ=2，σ²=1；eps=0、gamma=[1,1]、beta=[0,0]，输出 [-1,1]。练习 C：为何补齐到 8 个 lane 的 D=5 不能除以 8？因为 padding 不属于输入，均值定义的除数是有效元素数 D。

## 6. normalized_shape、行数和参数广播到底是什么

把输入视为 [B,D]，是把“需要一起归一化的后缀维度”折叠成 D，把外层维度折叠成 B。例如原输入 [2,3,4]：

| normalized_shape | 每组元素数 D | 独立组数 B | gamma/beta 的逻辑形状 |
|---|---:|---:|---|
| (4,) | 4 | 6 | [4] |
| (3,4) | 12 | 2 | [3,4] |

第一种会分别计算 6 组均值/方差；第二种只计算 2 组。它们输出形状虽然都为 [2,3,4]，结果却不同。新人常把“输出 shape 一样”误当成“归一化的轴一样”。

折叠维度还要求物理布局允许这么访问。本仓库 CUDA 示例只支持连续二维 FP32 行，不能把任意 transpose 后的 tensor 指针直接传进来。gamma/beta 每列对应不同参数，所有行共享同一组参数；它们不参与均值和方差的计算。

## 7. 给每个线程分任务，并对照真实 kernel

完整实现见 [row_ops.cu 的 row_layernorm](../../../assets/cuda/row_ops.cu)。它和 Softmax 复用同一个 block reduction helper，但归约内容不同：

~~~cpp
float local_sum = 0.0f;
for (int col = t; col < width; col += blockDim.x) {
  local_sum += x[base + col];
}
const float mean = block_reduce<false>(local_sum, scratch) / width;
~~~

这不是每线程求一个均值后再平均。每线程拿到的元素数可能不同，因此应先加总再除以真实 width。例如 D=5、T=4，线程 0 有两个元素，其余各一个；若先算四个线程的均值再直接除以 4，就给线程 0 的两个元素分配了错误权重。

第二阶段对原值减去同一个全行 mean：

~~~cpp
float local_m2 = 0.0f;
for (int col = t; col < width; col += blockDim.x) {
  const float diff = x[base + col] - mean;
  local_m2 += diff * diff;
}
const float variance = block_reduce<false>(local_m2, scratch) / width;
~~~

没有有效列的线程直接贡献 0，不能把补齐元素 x=0 也计算成 (0-mean)^2。只有参与原始输入的列才贡献平方差；分母仍是 width。

第三阶段计算 rsqrtf(variance+eps)，即平方根倒数，再对每个元素做减均值、乘倒数、乘 gamma、加 beta。使用倒数一次，再复用乘法，能减少逐元素重复开方；不同数学库与浮点路径的最后几位可能不同，因此必须用数值容差比较。

## 8. 一个“局部平均再平均”会失败的反例

取 x=[1,2,3,4,100]、T=4。线程 0 拿 [1,100]，局部均值为 50.5；其余均值为 2、3、4。直接平均这些局部均值得到 14.875，而真实均值为 110/5=22。

正确方法一是线程只输出局部 sum，最后除以总元素数。正确方法二是每线程保留 (count,mean)，合并时按 count 加权。这正是 Welford 状态里必须保存 count 的原因。

## 9. Welford 的逐元素更新：每个变量都有含义

维护三项状态：

~~~text
n    已处理元素数量
mean 已处理元素的均值
M2   Σ(x-mean)^2，尚未除以 n 的平方差和
~~~

读入新值 x 后：

~~~text
n_new    = n + 1
delta    = x - mean
mean_new = mean + delta/n_new
delta2   = x - mean_new
M2_new   = M2 + delta*delta2
~~~

两个 delta 不相同：第一个相对于旧均值，第二个相对于新均值。把乘积写成 delta² 是常见误改。

对 [1,2,3,4]，逐项状态如下：

| 新输入 | n | 新 mean | 新 M2 | M2/n |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | 0 |
| 2 | 2 | 1.5 | 0.5 | 0.25 |
| 3 | 3 | 2 | 2 | 0.666667 |
| 4 | 4 | 2.5 | 5 | 1.25 |

最终 M2=5，而方差是 1.25。把 M2 直接当成方差会让归一化尺度错误地多出 sqrt(D) 因子。

## 10. 两个 Welford 状态如何并行合并

GPU 不能只依赖单线程从头扫到尾，因此要让不同线程各自得到统计，再合并。令 A=(nA,meanA,M2A)、B=(nB,meanB,M2B)，delta=meanB-meanA：

~~~text
n     = nA+nB
mean  = meanA + delta*nB/n
M2    = M2A+M2B + delta²*nA*nB/n
~~~

最后一项来自“两个子集合的均值不同”。各自的 M2 只衡量对各自均值的偏差，把两堆数据放到共同均值下面，还会多出组间差异。

取 A=[1,2]，状态 (2,1.5,0.5)；B=[3,4]，状态 (2,3.5,0.5)。delta=2，总数 4，新均值 2.5；M2=0.5+0.5+4×2×2/4=5。若仅相加 M2A+M2B，结果只有 1，漏掉了组间差异。

空线程的统计为 n=0。合并函数先处理 nA=0 或 nB=0，返回另一边，避免除以零。运行：

~~~powershell
python assets/cpu/dataflow_lab.py --topic welford
~~~

应得到两组局部状态，以及合并后的 (4,2.5,5.0)，总体方差 1.25。实现位于 dataflow_lab.py 的 welford_state 和 merge_welford。CUDA baseline 为了教学清晰仍使用两遍统计；本章没有把它虚称为 Welford GPU 版。

## 11. 数值稳定、输入量化和算法误差要分开

E[x²]-E[x]² 的困难来自大数相减。以两个接近 10000 的值为例，平方和均值平方都在 10^8 附近，而方差可能只在 1 附近；有限精度中的有效位可能在相减前已丢失。两遍中心化或 Welford 通常更适合此类统计，但并不保证任意极端范围都不会溢出。

还有一种情况与统计公式无关：如果两个接近的真实值在转换为 FP16 时已经变成同一个数，后来用 FP32 甚至 FP64 累加也恢复不了原差异。测试 reference 应从“实际传入 kernel 的低精度输入值”转换后再算，避免拿原始未量化高精度值误判后端。

这三个误差来源应分别记录：输入存储精度、统计计算精度、最终输出舍入。随意加大容差会把索引 bug 和真实精度差异混在一起。

## 12. LayerNorm、BatchNorm、RMSNorm 为什么不能互换

LayerNorm 在每个样本的指定特征后缀内统计；BatchNorm 通常按通道在 batch/空间维上统计，训练和推理还涉及运行统计量；RMSNorm 按平方均值归一化，一般不减均值。接口参数和计算轴决定语义，三者并非只需换一个函数名。

例如 x=[2,2]，LayerNorm 的中心化部分是 [0,0]；RMS 归一化部分在 eps 很小时接近 [1,1]。这个例子已经足够证明它们不能直接替换。是否允许模型级替换属于模型设计问题，不能以 kernel 更快为理由自行改变。

## 13. 训练时反向传播多了什么

先只考虑一行，令 r=1/sqrt(var+eps)，z=(x-mean)*r，g 为传入的 dy，h=g*gamma。输入梯度可写为：

~~~text
dx = r/D × (D*h - sum(h) - z*sum(h*z))
~~~

gamma 的梯度是对所有外层行求 sum(dy*z)，beta 的梯度是 sum(dy)。这说明反向有两类不同归约：dx 在行内归约，dgamma/dbeta 沿行方向归约。直接把前向线程映射复制到反向不一定合适。

这组公式用于理解梯度数据流，不代表现有 CUDA 程序提供 autograd。若扩展训练，应使用小尺寸高精度有限差分验证 dx/dgamma/dbeta，再验证框架注册行为。opcheck 与梯度数学正确性是不同检查。

## 14. 运行、调试与验收

沿用 Softmax 章节的 row_ops 构建和运行命令，输出中查找 layernorm。常量行应该得到 beta；D=1 同样如此。对 gamma/beta 不同的列，输出错误若呈周期性重复，优先检查参数索引；若只有末尾错误，检查 mask；若所有值整体缩小，检查方差除数、eps 位置或误把 M2 当成 variance。

提交时同时附上两遍 CPU reference 与 Welford 合并的结果、D=1/常量/非整除宽度测试、接口支持范围和性能计时范围。练习答案不能只写“数值不稳定”，应说明丢失精度发生在哪一步。

## 参考资料（按学习顺序）

1. [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：核对方差、eps、normalized_shape 和 affine 语义。
2. [Triton LayerNorm Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)：学习行级映射和前后向。
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：并行归约与同步。
