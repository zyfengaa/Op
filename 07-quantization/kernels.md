# 量化 kernel：反量化放在哪、累加用什么

量化 kernel 的大部分设计决策，最后都收敛到三个问题：**反量化在哪一步做、累加器用什么 dtype、scale 在哪一层乘**。这三条决定了性能上限和数值正确性。算法侧的 scale 怎么来见 [algorithms.md](algorithms.md)，数值格式见 [basics.md](basics.md)。

## 反量化的三条路线

给定量化权重 `W_q` 和量化激活 `A_q`，反量化 `x = scale × (q − zero_point)` 可以在三个位置发生。

### 路线 ①：先整体反量化成 fp16，再做普通 GEMM

~~~text
读 int8 W_q (NK 字节) → 写 fp16 W (2NK 字节)
读 fp16 W (2NK 字节)  → 用普通 fp16 GEMM 计算
~~~

这是最省事、也最慢的做法。反量化是一次独立的逐元素 kernel，产生一个完整的 fp16 副本，走一遍完整的内存往返。

成本对比（以权重 W 的 N×K 个元素为单位）：

| 路线 | 权重相关访存 | 相对基线 |
|---|---|---|
| ① 整体反量化 | 读 1NK + 写 2NK + GEMM 再读 2NK = **5NK 字节** | 5× |
| ② 计算时反量化 | 只读 1NK，反量化在寄存器/shared memory 内 | 1× |
| ③ epilogue 融合 | 只读 1NK，只多一次 scale 乘法 | 1× |

**观察**：路线 ① 的权重访存是另外两条的约 5 倍。
**假设**：如果原本就受计算限制，这多出的带宽可能被隐藏；如果受带宽限制，路线 ① 会直接变慢。是否隐藏要实测。

### 路线 ②：在计算时反量化

把量化权重搬进 shared memory 或寄存器后，在参与乘前的最后一步反量化成计算 dtype。没有全量 fp16 副本，代价是每条数据路径多几条指令（乘 scale、减 zero_point）。适合**权重 only 量化**（W4A16 / W8A16）：激活是 fp16，权重是 int4/int8，反量化只发生在权重一侧。

### 路线 ③：把反量化融进 GEMM 的 epilogue

当计算本身就是 int8×int8 时，累加得到 int32，最后再乘 `scale_a × scale_b` 得到浮点输出。反量化退化成 epilogue 里的**一次乘法**，没有额外的反量化 kernel，也没有逐元素的独立循环。这是 W8A8 的标准做法。

**选择判据**：

| 场景 | 推荐路线 | 原因 |
|---|---|---|
| W8A8，激活也量化 | ③ | 累加是整数，反量化就是 epilogue 的 scale 乘法 |
| W4A16 / W8A16，仅权重量化 | ② | 激活是浮点，没有整数累加可融合，只能在权重侧就地反量化 |
| 原型验证、先跑通 | ① | 实现最简单，但不要把它当成最终性能 |

## int8 GEMM 的累加：为什么必须是 int32

int8 乘法的最大乘积是 `127 × 127 = 16129`，能放进 int16（最大 32767）。但**只有一项时**成立：两项就到 32258，三项就溢出。

~~~text
int8 × int8 单项最大值      = 127 × 127 = 16129
int16 最大值                = 32767        → K = 2 就溢出
int32 最大值                = 2147483647   → K 上界 ≈ 2^31 / 16129 ≈ 133000
~~~

所以标准做法是 **int8 输入、int32 累加**。它有两条与浮点本质不同的性质：

- **整数累加是精确的**：只要不溢出，中间结果没有舍入误差。误差不随 K 增长。
- **唯一风险是溢出**：K 超过约 1.3e5 且乘积同号时，int32 会绕回。超长 K（如大词表投影）要显式评估。

**结论**：int8 GEMM 的误差**几乎全部来自输入量化本身**，不来自累加。这与 fp16 累加（误差随 K 增长）完全不同，评估时的关注点也不同——见 [evaluation.md](evaluation.md)。

对称量化时 `zero_point = 0`，累加干净。非对称（激活 uint8）时存在 zero_point 交叉项：

~~~text
A_q ∈ [0,255],  W_q ∈ [-128,127],  zero_point = z
sum_k (A_q[k]−z)·W_q[k] = sum_k A_q[k]·W_q[k] − z · sum_k W_q[k]
~~~

第二项只依赖权重的行和，可以**预计算**，在 epilogue 里一次减掉。不要在 K 循环里逐项减 zero_point，那会把整数累加变成带修正的路径，得不偿失。

## per-channel / per-group scale 对 kernel 的影响

| 粒度 | scale 形状 | 加载与广播方式 | 访存影响 |
|---|---|---|---|
| per-tensor | 标量 | 1 个寄存器，全程广播 | 无额外开销 |
| per-channel | 长度 N 的向量 | epilogue 按输出列取对应 scale；N-tile 与向量对齐 | 权重按 [N,K] 行量化，加载无额外成本 |
| per-group | [N, K/G] 的矩阵 | K 循环内随权重 tile 一起加载 scale tile | 组大小与 K-tile 对齐时规律；不对齐则额外加载和分支 |

per-group（int4 常见）的关键是**让 K-tile 与 G 成整数倍关系**：

~~~text
推荐：G = 128，K-tile = 64（一个 group 跨 2 个 k 迭代，加载规律）
      G = 128，K-tile = 128（一一对应，最干净）
避免：G = 128，K-tile = 192（1.5 个 group，半组跨 tile，掩码和分支进入热路径）
~~~

**观察**：组粒度不对齐时，scale 的加载量可能和权重本身同量级，把量化省下的带宽又吃掉。
**假设**：具体损失取决于硬件和 tile 尺寸，用 profiler 看访存事务数验证，不要凭组大小推断。

## 融合顺序：scale 必须留在 epilogue

这是量化 kernel 最容易犯的错：**把 scale 乘在数据路径中间**。

~~~text
错误：反量化每个操作数 → 乘 → 用低精度累加
正确：整数/低精度乘 → 高精度累加 → epilogue 里乘一次总 scale
~~~

用一个三轮点积说明差别：

~~~text
a_q = [100, 100, 100], scale_a = 1/300
b_q = [100, 100, 100], scale_b = 1/300
真值 dot = 3 × (1/3)² = 1/3 ≈ 0.3333333

路线 A（累加完再乘，正确）
  acc_int32 = 3 × 100 × 100 = 30000        （整数，精确）
  scale = scale_a × scale_b = 1/90000
  result = 30000 / 90000 = 0.3333333
  误差：仅末次 fp32 舍入，相对误差 ≤ 2^-24 ≈ 6e-8

路线 B（先把每个操作数反量化成 fp16 再累加）
  a_fp16 = round_fp16(1/3) ≈ 0.333252      （相对误差约 2.4e-4）
  每项 ≈ 0.111084，三项和 ≈ 0.333252
  误差：相对约 2.4e-4
~~~

路线 B 因为**在每个操作数上各舍入一次**，相对误差比路线 A 大约四个数量级。K 越大，路线 B 还会再叠加累加舍入。

**除了精度，还有量级问题**：如果把 scale 乘在路径中间，累加器里装的是**已经缩放过的值**。若 `scale_a × scale_b` 很小，值下溢进 denormal 甚至 0；若很大，可能溢出。路线 A 的累加器装的是整数，量级可控，缩放只在最后发生一次。

**与 [03-operators/matmul/walkthrough.md](../03-operators/matmul/walkthrough.md) 的 tiling/epilogue 完全对应**：K 循环里的寄存器 `acc` 就是累加器——量化版本里它是 **int32**，而不是 fp16/fp32。K 循环结束后才进入 epilogue：乘 `scale_a × scale_b`、加 bias、套激活。把 scale 乘在 K 循环内部，等于把 epilogue 的活提前塞进数据路径，既丢精度又涨指令数。这与 [融合页](../04-performance/techniques/fusion.md) 的原则一致：融合要放在**归约之后的 epilogue**，不能进归约内部。

## 常见误区

**把反量化当成普通逐元素算子，先整体转 fp16。** 多一次全量内存往返，权重访存约 5 倍。除非在原型阶段，不要用路线 ①。

**在错误的粒度上融合。** 权重量化用了 per-channel，kernel 却按 per-tensor 乘 scale，等于把通道差异抹平——结果“能跑但精度莫名下降”。scale 的粒度和权重的量化粒度必须一致。

**忽略累加的 dtype。** int8 用 int16 累加，K 稍大就静默溢出；int8 累加进了 fp16，误差随 K 增长。累加 dtype 属于契约，必须显式写明。见 [basics.md](basics.md)。

**把 scale 乘在数据路径中间。** 见上，精度差四个数量级，还可能下溢/溢出。

**K 循环里逐项处理 zero_point。** 非对称的 zero_point 补偿项应预计算，在 epilogue 一次减掉。

**per-group 组大小随便定。** 组大小与 K-tile 不对齐会把掩码和分支带进热路径，省下的带宽又还回去。

## 去哪查

- [GEMM 完整案例](../03-operators/matmul/walkthrough.md)：tiling、寄存器累加器与 epilogue 的结构
- [GEMM 契约](../03-operators/matmul/README.md)：shape、stride、累加类型必须声明
- [融合](../04-performance/techniques/fusion.md)：epilogue 融合的通用判据
- [低精度基础](basics.md)：累加器 dtype 与误差累积
- [精度评估](evaluation.md)：怎么验证上面的数值结论
