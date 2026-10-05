# 手法 · Tensor Core 与 MMA

**适用瓶颈**：算术强度高于机器平衡点、计算受限，且计算主体是矩阵乘（或可映射为矩阵乘的卷积、注意力）——[阶段五](../../01-workflow/05-locate.md)判定为 Compute Bound 且有效算力远低于对应 dtype 的峰值。

## 症状

前三条是**观察**，后两条是待验证的**假设**：

- 观察：有效算力远低于峰值，但算术强度高于机器平衡点（瓶颈不是带宽）。
- 观察：时间线上计算端口空置，stall 以 `not selected` / `wait` 或发射相关为主，而不是 `long scoreboard`。
- 观察：SASS/PTX 里是 `FFMA`（标量乘加），没有 `HMMA` / `IMMA` / `wgmma` / `mma.sync`。
- 假设：把计算改写为 tile 级矩阵乘加后，算力利用率上升。
- 假设：M、N、K 的尺寸足以让 tile 形状与 MMA 指令对齐，边界只占很小比例。

**先算平衡点**：`机器平衡点 = 峰值算力 / 峰值带宽`。AI 低于它时是内存受限，追求 Tensor Core 利用率没有意义（见「什么时候不要做」）。这一步决定方向，不能跳。

## 手段

### 1. 用 MMA 指令替换逐元素 FMA

**MMA 是 warp（或 warpgroup）协同执行的固定形状矩阵乘加指令，不是一个线程算一个结果。** 整组线程共同持有一个 tile 的 A、B、C 片段，一次指令完成 `D = A×B + C`。

| 指令 | 形状 (M×N×K) | 参与线程 | 输入 dtype |
|---|---|---|---|
| `mma.sync` | 16×8×8 | 1 warp（32） | tf32 |
| `mma.sync` | 16×8×16 | 1 warp（32） | fp16 / bf16 |
| `mma.sync` | 16×8×32 | 1 warp（32） | int8 / fp8 |
| `wgmma`（sm_90） | 64×N×16 / 64×N×8 | warpgroup（128） | fp16/bf16 / tf32 |

**成立条件**：

- 计算可写成 `D = A×B + C` 的 tile 形式，或能通过 im2col 等变换映射过去。
- block 内的 tile 形状按指令形状切分；M、N、K 的边界用 padding 或尾块路径处理。
- 输入 dtype 与目标指令匹配（见手段 2）。

### 2. 数据类型匹配：FP16 输入 ≠ 用了 Tensor Core

fp16/bf16 **存储**不等于 fp16/bf16 **矩阵指令**。若变量是半精度但仍用 `FFMA` 计算，走的是 SIMT 路径，峰值是标量算力，与矩阵指令的峰值差一个量级。

**成立条件（必须验证，不能假设）**：查 SASS 中是否出现 `HMMA` / `IMMA` / `OMMA` / `QMMA`，或 PTX 中是否出现 `mma.sync` / `wgmma`。只有看到这些指令，才能声称用了 Tensor Core。本库反复强调这一点：**dtype 是输入格式，Tensor Core 是指令路径，两者独立。**

### 3. TF32 的默认行为对数值的影响

TF32 是 19 位格式（8 位指数 + 10 位尾数），用于 fp32 矩阵乘的 Tensor Core 路径。**它默认是否开启取决于框架与版本**（例如 PyTorch 对 matmul 的 `allow_tf32` 默认值在版本间变更过）。

**成立条件**：

- 契约要求 fp32 精度时：必须**显式关闭** TF32，并用日志或 SASS 确认实际走的路径。
- 契约允许降低精度换速度时：明确写进契约，并在容差内验证。
- 这是**契约问题，不是性能问题**：开启 TF32 会改变结果，不能事后声称「只是优化」。

### 4. 小 M 的 decode GEMM 为什么用不上

decode 逐 token 时 M 很小（常为 1 到几），权重每个只被读一次，算术强度近似 `2M/s`（s 为每个权重的字节数）：M=1、fp32 时约 0.5 FLOP/byte，fp16 时约 1。**远低于机器平衡点，瓶颈是权重流与延迟隐藏，不是 Tensor Core 利用率。**

**成立条件（反向）**：只有当 M 增大到让同一权重服务足够多的输入行、AI 越过平衡点时，tile 复用策略才重新有效。这个判断要在具体设备上用解码 GEMM 的实测验证，不能照搬大 GEMM 结论（见 [decode-gemm](../../03-operators/matmul/decode-gemm.md)）。

## 什么时候不要做

- **memory bound 的负载上追求 Tensor Core 利用率。** 有效带宽已接近峰值、AI 低于平衡点时，算力不是瓶颈，提高利用率不改时间。
- **在 M 极小的 decode 场景套用大 GEMM 的 tile 策略。** 大 M tile 会大片是 padding 或空转，且 A 行没有可复用的对象；先解决权重流与延迟隐藏。
- **输入 dtype 与目标指令不匹配却强转。** 转换开销与精度损失可能超过算力收益；转换本身也走带宽。
- **精度契约不允许降级时用低精度累加。** 累加器 dtype 属于契约（见 [dtype 与精度](../../02-foundations/contracts/dtype-and-precision.md)）。
- **只对齐 M、N 不对齐 K。** K 边界需按指令的 K 补齐或走尾块，否则正确性先崩。

## 验证方式

| 指标 | 期望变化 |
|---|---|
| SASS/PTX 中 MMA 指令数 | 从 0 变为 > 0（**指令级直接证据**） |
| Tensor pipe 利用率 | 上升 |
| 有效算力 | 上升，逐步逼近该 dtype 的峰值 |
| kernel 时间 | 下降（最终判据） |
| 数值误差 vs fp32 参考 | 在容差内（记录 TF32/低精度的影响） |
| decode 小 M 场景 | 若有效带宽上升而非算力上升，说明方向是访存，不是 Tensor Core |

**注意**：某 dtype 的峰值与另一种 dtype 的峰值不可比。不能用 fp32 的宣传峰值评判 fp16 路径，也不能反过来。

## 常见误区

**「输入是 fp16/bf16 就是 Tensor Core」。** 不是。要确认生成了 MMA 指令，本库反复强调。

**拿 fp32 的峰值评判 fp16 路径。** 两者是不同的峰值，混用会得出错误利用率。

**开着 TF32 却要求 fp32 精度。** 结果会变，属于契约错误。

**在 decode 小 M 上宣称提升 Tensor Core 利用率。** 瓶颈在权重流和延迟隐藏，方向性错误。

**只切 M、N，不对齐 K。** 尾块与对齐问题先导致正确性失败。

**把小 shape 的 tile 策略直接推广到大 shape（或反之）。** tile 形状要按实测的有效算力与占用率重新选取。

## 相关页面

- [性能域索引](../README.md)：手法与瓶颈的映射表
- [瓶颈分析](../analysis.md)：算术强度与机器平衡点
- [阶段五 · 定位](../../01-workflow/05-locate.md)：如何区分计算受限与内存受限
- [GEMM 案例](../../03-operators/matmul/README.md)：从 SIMT 到 MMA 的版本路线
- [decode GEMM](../../03-operators/matmul/decode-gemm.md)：M=1 时为什么用不上大 GEMM 策略
- [dtype 与精度](../../02-foundations/contracts/dtype-and-precision.md)：TF32 与累加器 dtype
- [occupancy](occupancy.md)：tile 变大后的寄存器与 shared 压力
- [术语表 · MMA / TF32](../../99-reference/glossary.md)
