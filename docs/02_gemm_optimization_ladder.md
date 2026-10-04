# GEMM 优化阶梯实验

这一章把一个 GEMM 拆成多个可比较的版本。每一步只引入一个主要概念，并保留数学公式、线程/数据映射、可能的瓶颈、要观察的指标和完成标准。

贯穿示例：`C[M,N] = A[M,K] × B[K,N]`，`M=N=K=4096`，`dtype=FP16`。教学时还必须测试 `M=4103,N=4099,K=4097`，因为只在整除 tile 的尺寸上正确不算完成。

## Stage 0：数学和 CPU reference

先写最清晰的三重循环：

```text
for m in M:
  for n in N:
    acc = 0
    for k in K:
      acc += A[m,k] * B[k,n]
    C[m,n] = acc
```

这一版的价值是定义答案，不是性能。验证 `2MNK` 次浮点操作、输出 shape 和误差阈值。所有 GPU 版本都必须和它对比。

## Stage 1：naive GPU GEMM

让一个线程负责一个 `C[m,n]`：

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

它容易理解，但不同线程会重复从 Global Memory 读取 A/B。要观察 Kernel 时间、有效带宽、Global Load Efficiency 和边界正确性。

## Stage 2：修复 coalescing

分析一个 warp 中相邻线程访问的地址。如果线程沿连续列访问 row-major 数据，相邻线程更容易形成合并访问；如果线程跨行访问，地址跨度可能达到 `K × sizeof(dtype)`。

固定数学结果，只改变线程映射，比较 requested bytes、actual traffic 和 Global Load Efficiency。不要因为 HBM Throughput 上升就断言更快，要看有效带宽和实际时间。

## Stage 3：Shared Memory Tiling

使用 `BM=128, BN=128, BK=32` 的概念 tile：A tile 为 `128×32`，B tile 为 `32×128`，输出 tile 为 `128×128`。一个 tile 的 A/B 数据加载一次后，被多个线程重复使用，减少 Global Memory 访问。

任意尺寸必须使用 ceil grid：

```text
grid_m = (M + BM - 1) / BM
grid_n = (N + BN - 1) / BN
```

加载和写回都要做边界检查，越界加载可补零，越界结果不能写回。每个 K tile 都要同步：加载完成后才能计算，计算完成后才能覆盖 shared memory。

## Stage 4：Register Tiling

让一个线程保留小块 accumulator，提升寄存器内的数据复用，减少 shared memory 往返。但寄存器过多会降低 resident block 数量。保持 tile 不变，只修改每线程 accumulator 数量，记录 `registers/thread、occupancy、Kernel time`。你可能会看到 occupancy 降低但时间变快，这说明 occupancy 不是唯一目标。

## Stage 5：Tensor Core / MMA

FP16/BF16 只是可能使用矩阵指令的前提，不是充分条件。要检查 shape 是否适配 MMA tile、K 是否满足指令要求、layout 是否匹配、编译器是否真正发出 MMA 指令，以及 warp 是否有足够连续的 MMA 工作。

对于 `M=4103,N=4099,K=4097`，尾块需要 mask、padding 或 fallback。规整 shape 上 Tensor Core 利用率高，不能证明任意 shape 都高。

## Stage 6：Double Buffer 和 Pipeline

无 pipeline 的时间线近似为 `LOAD0 → WAIT → MMA0 → WAIT → LOAD1 → WAIT → MMA1`。double buffer 目标是让 `MMA0 + LOAD1` 重叠，再让 `MMA1 + LOAD2` 重叠。

真实实现会涉及 async copy、同步和 buffer ownership。必须先确认瓶颈是 load latency 或供数不连续；如果 HBM 已接近饱和，double buffer 不能突破带宽上限。

## Stage 7：模拟一次 profiling 决策

初始结果：

```text
Time                  1.00 ms
Tensor Core Util      38%
SM Util                72%
Occupancy              25%
HBM BW                 41%
L2 Hit Rate            63%
Long Scoreboard Stall  高
```

分析顺序：

1. `M=N=K=4096` 且 tile 可整除，Shape 不是首要问题。
2. FP16 且确认存在 MMA 指令，dtype/path 基本正确。
3. AI 高，理论偏 Compute Bound；但 HBM 和 Tensor Core 都没满，不像单纯算力上限。
4. `Registers/thread=160` 导致 Occupancy 只有 25%，先缩小 accumulator/register tile。
5. 若 `160→112` 后 `25%→50%` 且时间 `1.00→0.78 ms`，说明寄存器压力假设成立。
6. 若 shared load bank conflict 很高，修改 padding 或 swizzle；若时间 `0.78→0.66 ms`，说明 shared layout 假设成立。
7. 若 MMA 之间仍有 wait，加入 double buffer/async copy；若时间 `0.66→0.49 ms`，说明 pipeline 假设成立。

最终可得到 `1.00 ms → 0.49 ms ≈ 2.04x`。这些数字仍然是教学模拟值；真正项目必须提供 profiler 原始报告和重复测量结果。

## Stage 8：每一步的验收清单

| 检查项 | 最低要求 |
|---|---|
| 正确性 | 与 CPU reference 对比，覆盖规整和非规整 shape |
| 数值 | 明确 dtype、累加类型和误差阈值 |
| 性能 | 固定 warmup、重复次数和输入，报告平均值/分位数 |
| 分析 | 写出至少一条“观察 → 假设 → 修改 → 验证” |
| 资源 | 记录 registers/thread、shared memory、occupancy |
| 访存 | 检查 coalescing、transaction、bank conflict、traffic |
| 计算 | 确认实际指令路径和 Tensor Core/MMA 发射情况 |
| 回归 | 测试小尺寸、边界尺寸、非整除尺寸和大尺寸 |
| E2E | 证明算子级收益没有被框架或其他阶段抵消 |

## Stage 9：从 GEMM 迁移到其他算子

```text
Vector Add → ReLU → Reduce → Transpose → Softmax
→ Matmul → LayerNorm → 融合算子
```

迁移时复用接口契约、CPU reference、shape/stride 分析、并行分解、内存层级、性能假设和验证闭环；不要把某个 CUDA warp 宽度、某条 MMA 指令或某个 profiler 指标当成通用规律。
