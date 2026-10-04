# Decode 小 M GEMM：一次完整的算子性能分析

## 1. 场景和目标

假设从一个 Qwen 类模型的 vLLM Decode profiling 中看到：某个 GEMM 占单 token 端到端耗时的四分之一。我们的目标不是立刻重写 Kernel，而是回答：

1. 这个 Kernel 对应模型中的哪一步？
2. 它为什么慢，属于哪一种瓶颈？
3. 哪个优化方向最值得先做？
4. 优化后如何证明不是“指标变好看了”，而是真正变快了？

本文所有数字都是模拟数据，用于演示推理过程。

## 2. 先锁定问题对象

模拟 profiling：

```text
Decode E2E / token = 320 us
GEMM Kernel        = 82 us
Kernel 占比        = 25.6%
Tensor Core Util    = 8%
HBM BW Util        = 72%
Occupancy          = 31%
L2 Hit Rate        = 18%
```

正确的第一反应不是“优化 GEMM”，而是沿调用链确认它是谁：

```text
vLLM
  → Qwen Decoder Layer
  → MLP / Attention Projection
  → Linear
  → matmul
  → Vendor GEMM
  → gemm_1x4096x4096
```

最终确认本次工作负载为：

```text
X: [1, 4096]      FP16
W: [4096, 4096]   FP16，row-major
Y: [1, 4096]      FP16

Y = X @ W
M = 1
N = 4096
K = 4096
```

`M=1` 是关键线索：Decode 每次通常只处理一个新 token，矩阵在 M 方向几乎没有并行空间。它更接近“很多输出元素各自做一次向量点积”的 GEMV 型工作负载，而不是适合大规模 CTA 铺开的传统大 GEMM。

## 3. Shape：先判断计算形态

矩阵乘法为：

```text
A[M, K] × B[K, N] = C[M, N]
```

理论计算量：

```text
FLOPs = 2 × M × N × K
      = 2 × 1 × 4096 × 4096
      ≈ 33.6 MFLOPs
```

如果 `M=N=K=4096`，GPU 可以在 M、N 两个方向铺出大量 CTA；但在 `M=1` 时，M 方向只有一行。常规大 GEMM 的 tile 可能严重浪费：例如 M tile 为 128，实际只有 1 行有效数据，M 方向有效利用率约为 `1/128`。

因此，初步假设是：

```text
Small-M
→ 并行度下降
→ 数据复用下降
→ Tensor Core 不容易吃满
→ 优先怀疑 Memory / Latency Bound
```

这只是工作假设，不能仅凭 shape 下结论。

## 4. Dtype 和访存量：权重是主角

FP16 每个元素占 2 Byte：

```text
权重：4096 × 4096 × 2 ≈ 32 MB
输入：1 × 4096 × 2    ≈ 8 KB
输出：1 × 4096 × 2    ≈ 8 KB
```

为了完成约 33.6 MFLOPs 的计算，需要扫描约 32 MB 权重。输入和输出相对很小，Kernel 的数据移动主要由权重决定。这会把问题推向内存系统，而不是单纯的峰值计算能力。

## 5. Arithmetic Intensity 和 Roofline

算术强度定义为：

```text
Arithmetic Intensity = FLOPs / Memory Bytes
```

近似只按权重读取计算：

```text
AI ≈ 33.6 MFLOPs / 32 MB
   ≈ 1 FLOP / Byte
```

假设 GPU FP16 峰值为 100 TFLOPS、HBM 峰值带宽为 1.5 TB/s，则 Roofline 分界点为：

```text
100 TFLOPS / 1.5 TB/s ≈ 66.7 FLOP / Byte
```

工作负载的 `AI≈1` 远低于分界点，理论上位于 Memory Bound 区域：

```text
AI 低 → 能达到的性能受带宽限制
AI 高 → 才更可能受计算吞吐限制
```

所以 `Tensor Core Util=8%` 更可能是 Memory Bound 的结果，而不是第一优先级的根因。正确问题是：“为什么数据没有高效、连续、足够并行地送到计算单元？”

## 6. 用 profiler 验证，而不是停留在猜测

### 6.1 HBM 带宽

如果 HBM 利用率为 72%，理论有效带宽约为：

```text
1.5 TB/s × 72% ≈ 1.08 TB/s
```

32 MB 权重的理想流式读取时间约为 `32 MB / 1.08 TB/s ≈ 29.6 us`，但实际 Kernel 为 82 us。这说明它不是一次理想的连续 HBM streaming，还存在访存事务浪费、低并发、延迟未隐藏或 Kernel 固定开销。

### 6.2 Layout 和 coalescing

对于 row-major 的 `W[K,N]`，理想情况下一个 warp 的相邻线程访问同一行的相邻列：

```text
lane0 → W[k][0]
lane1 → W[k][1]
lane2 → W[k][2]
```

如果线程映射变成按列跨行访问：

```text
lane0 → W[0][n]
lane1 → W[1][n]
lane2 → W[2][n]
```

相邻线程地址相差约 `4096 × 2 Byte`，会产生离散的 memory transaction。应检查：

- global load efficiency；
- requested bytes 与 actual DRAM traffic；
- memory transaction / sector utilization；
- L2、DRAM request 数量。

例如 requested bytes 为 32 MB、actual DRAM traffic 为 58 MB，说明实际算术强度还低于理论估算，应先修复线程映射和合并访存。

### 6.3 Occupancy 和寄存器压力

`Occupancy=31%` 不能直接等价为“必须提高 Occupancy”。先查原因：

```text
Registers / thread
Shared memory / CTA
Threads / CTA
CTA 数量是否足够
```

若 `Registers/thread=128`，一个 SM 可能只能驻留 2 个 CTA 而不是 8 个。内存访问发起后，缺少可切换的 warp 就不能隐藏 HBM 延迟；如果同时看到 `Long Scoreboard Stall=42%`，就形成了较完整的证据链：

```text
寄存器压力
  → 可驻留 CTA 减少
  → 无法隐藏内存延迟
  → Long Scoreboard Stall 增加
  → 带宽利用率只有 72%
```

### 6.4 L2 Hit Rate

`L2 Hit=18%` 不一定是异常。模型权重通常远大于 L2，而 M=1 时每个权重往往只读取和使用一次，天然没有足够的数据复用。此时花大量时间“提高 L2 命中率”可能没有收益。

性能分析的一个重要能力，是判断哪些指标不值得优化：指标异常不代表它就是根因，也不代表存在可利用的复用机会。

## 7. 重新定义优化方案

到这里，优化目标应从“提高 Tensor Core 利用率”改成：

```text
Small-M GEMV specialization
  + 合理的 N 方向 tile
  + coalesced / vectorized load
  + 降低 register pressure
  + 提高足以隐藏延迟的并发
  + 必要时做 load/compute pipeline
```

每一项都需要单独验证。不要一次改完所有内容，否则即使变快，也无法知道哪条假设成立。

Pipeline 可以把下一批数据加载和当前计算重叠，但它只能隐藏延迟，不能突破 HBM 带宽上限。如果带宽已经接近 95%，继续堆 double buffer 通常收益有限；本例带宽只有 72%，且 scoreboard stall 较高，因此 pipeline 有合理依据。

## 8. 第二阶段：考虑融合

假设 GEMM 优化后为 48 us，但调用链仍为：

```text
GEMM          48 us
Bias           8 us
SiLU          11 us
Elementwise    7 us
```

如果每个中间结果都写回并再次从 HBM 读取，Memory Bound 工作负载会承担大量额外流量。可以在 accumulator 上直接完成：

```text
accumulator
  → + bias
  → SiLU
  → 最终写回
```

这就是 GEMM + Bias + Activation fusion。融合的价值不在于让某个 Tensor Core 指标更高，而在于减少中间结果的 HBM read/write。

## 9. 用数据闭环证明优化

模拟的分阶段结果：

| 版本 | Kernel 时间 |
|---|---:|
| Vendor GEMM | 82 us |
| GEMV specialization | 61 us |
| + vectorized load | 54 us |
| + register tuning | 49 us |
| + pipeline | 43 us |

算子级速度提升为 `82/43≈1.91x`。指标变化可以是：

| 指标 | Before | After |
|---|---:|---:|
| HBM BW | 72% | 89% |
| Occupancy | 31% | 62% |
| Long Scoreboard | 42% | 17% |
| Memory efficiency | 63% | 91% |
| Tensor Core Util | 8% | 12% |

这里最值得注意的是 Tensor Core 只从 8% 变成 12%，但 Kernel 从 82 us 降到 43 us。它说明 Tensor Core 利用率不是该 workload 的核心 KPI，不能拿一个漂亮的指标替代真实性能。

最后必须看 E2E：

```text
原始 Decode E2E：320 us/token
仅优化 GEMM：  281 us/token，约 12.2% 提升
再做融合：     263 us/token，约 17.8% 提升
```

端到端收益小于单 Kernel 收益是正常的，因为其他 Kernel、KV Cache、调度、通信和框架开销仍然存在。只有 E2E 变快，才说明这次优化对用户可见性能有价值。

## 10. 可复用的标准工作流

面对任何 Top Kernel，都可以按以下顺序工作：

```text
定位 Top Kernel
  → 映射模型结构
  → 确认 M/N/K、dtype、layout
  → 计算 FLOPs 和 Memory Bytes
  → 计算 Arithmetic Intensity
  → 用 Roofline 初步定性
  → 检查带宽、SIMD/Tensor Core、Occupancy、Cache
  → 检查访存合并、寄存器、Pipeline、Launch
  → 建立“指标 → 根因”证据链
  → 提出一个可验证的优化假设
  → 实现并做 Correctness
  → Micro Benchmark
  → Profiler 验证假设
  → E2E Benchmark
```

最重要的不是每一项都看过，而是能写出因果链。例如：

```text
M=1
  → 数据复用低
  → Arithmetic Intensity≈1
  → Memory Bound
  → Tensor Core 低不是核心问题
  → 带宽只有 72%
  → 继续检查 memory access 和 register pressure
  → Occupancy 低、scoreboard stall 高
  → 选择 GEMV、vector load、register tuning、pipeline 和 fusion
```

AI 可以快速生成多个 Kernel 版本，但不能替你决定“为什么此时应该优化 vector load，而不是 Tensor Core、L2 或 K tile”。这个判断，以及最后用数据证明判断，是算子开发在 AI 时代仍然最有价值的能力。

## 本章参考资料

- [CUDA C++ Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
