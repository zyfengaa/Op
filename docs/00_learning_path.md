# 从零开始的算子开发学习路线

本文面向写过业务代码、模型推理或普通 C/C++/Python，但没有系统做过算子开发的开发人员。目标不是背 CUDA API，而是建立一套可以迁移到不同 GPU/NPU 的工作方法。

## 1. 先理解“算子开发”在解决什么问题

一个算子至少包含四层含义：

```text
数学语义：输入是什么，输出是什么，公式是什么？
接口契约：shape、dtype、layout、stride、设备和错误边界是什么？
并行实现：哪些元素能并行，线程/warp/block 如何映射？
性能证据：为什么快，瓶颈在哪里，优化是否真的改善了 E2E？
```

因此，“代码能运行”只证明数学语义可能正确；“Kernel 很复杂”也不证明它很快。算子开发的完整交付物应包括 reference、测试、benchmark、profiling 证据和限制条件。

## 2. 总路线和每阶段交付物

| 阶段 | 学习目标 | 必做小案例 | 交付物 |
|---|---|---|---|
| 0. 基础 | 看懂矩阵、指针、shape、dtype 和误差 | Vector Add、ReLU、矩阵加法 | 公式、接口和边界说明 |
| 1. 并行思维 | 从元素依赖推导线程映射 | 1D/2D indexing、任意尺寸处理 | CPU reference、正确性测试 |
| 2. GEMM 入门 | 理解 M/N/K 和计算量 | naive GEMM、非整除 shape | 最小实现、输入输出样例 |
| 3. 内存系统 | 理解合并访存、缓存和复用 | row/column access、tile 复用 | 访存示意、带宽计算 |
| 4. Tiling | 把 Global Memory 数据搬到更快层级 | shared-memory tiled GEMM | tile 设计和边界策略 |
| 5. 线程资源 | 理解 warp、register、shared memory、occupancy | register tile、不同 block size | 资源假设和 profiler 指标 |
| 6. 矩阵指令 | 理解 MMA/Tensor Core 的约束 | FP16/BF16 MMA 路径 | dtype、layout、指令路径验证 |
| 7. Pipeline | 让 load 和 compute 重叠 | double buffer、async copy | 时间线和 stall 分析 |
| 8. 工程闭环 | 让结果可复现、可集成、可回归 | benchmark、误差、E2E | 报告、回归测试、文档 |
| 9. 跨平台 | 区分通用方法和后端特性 | CUDA 与其他后端对照 | 公共层/后端层边界 |

不要跳过前面的阶段直接学 Tensor Core。若 shape、layout、边界或 reference 都没有定义，后面的高性能指令只会让错误更难定位。

## 3. 每章固定学习模板

每个新算子都按同一张检查表推进：

### 3.1 定义问题

- 公式是什么？例如 `C[M,N] = A[M,K] × B[K,N]`。
- 输入输出 shape、dtype、layout、stride 是什么？
- 是否支持任意尺寸，还是只支持特定对齐？
- 数值误差标准是什么？FP32、FP16、BF16 不能用同一套直觉判断。

### 3.2 写 reference

先写一个最容易读懂的 CPU reference。它不追求快，只负责定义正确答案。所有后续 Kernel 都与 reference 对比，至少覆盖：空输入、1 元素、非 tile 整除尺寸、较大尺寸和随机输入。

### 3.3 建立 baseline

记录一次可复现的 baseline：输入 shape、dtype、设备、编译参数、warmup 次数、测量次数、平均值/分位数。没有 baseline 就没有“优化前后”。

### 3.4 建立性能假设

不要写“感觉 shared memory 会更快”，而要写成可以被 profiler 证伪的假设：

```text
观察：global load efficiency 低，actual traffic 明显大于 requested bytes。
假设：线程沿错误维度访问，访存没有合并。
修改：交换线程映射或改变 layout。
验证：效率上升，actual traffic 下降，Kernel 时间下降。
```

### 3.5 一次只改一个主要变量

先改 tile，再改线程映射，再改寄存器或 pipeline。一次把所有优化叠加，结果即使变快也无法知道哪条假设成立；学习和工程排障都应该保留逐阶段版本。

## 4. 必须掌握的九个概念

### Shape：工作量和并行度

GEMM 的计算量是 `2MNK`，但 shape 还决定 CTA 数量、尾块数量和 GPU 是否有足够工作可调度。`M=N=K=4096` 与 `M=17,N=4096,K=4096` 的 FLOPs 可能都很大，但后者 M 方向只能产生很少的 CTA，优化重点可能是并行度而不是 tile 内部计算。

### dtype：数据大小和计算路径

dtype 同时影响内存流量、寄存器/共享内存占用、累加精度以及能否走 FP16/BF16/FP8 的矩阵指令。看到 FP16 不能直接断言使用了 Tensor Core，必须检查实际指令路径。

### layout：地址如何变化

连续线程访问连续地址，通常更容易形成 coalesced access。若相邻线程地址间隔为 `K × sizeof(dtype)`，会产生大量 memory transaction；“显存很忙”不等于“数据搬得有效”。

### Arithmetic Intensity：先做理论定性

```text
AI = FLOPs / Bytes
```

AI 低时优先调查 bandwidth、layout、复用和融合；AI 高时再重点调查 Tensor Core、指令吞吐、tile 和 pipeline。Roofline 是初始分类，不是替代真实 profiler 的结论。

### Tile 和 Warp Mapping：决定谁加载、谁计算

tile 把 A/B 的一小块搬到更快的内存层级，多个线程重复使用。tile 大小又会影响共享内存、寄存器、CTA 数量和尾块浪费；不存在对所有 shape 都最优的固定 tile。

### Occupancy：不是越高越好

Occupancy 是活跃 warp 与硬件最大 warp 的关系。寄存器太多、shared memory 太大或 block 太重都会降低它，但过度压低寄存器也可能减少数据复用和指令级并行。目标是“足以隐藏延迟，并持续供给计算单元”，不是盲目最大化百分比。

### Cache 和内存层级：问数据有没有复用

```text
HBM → L2 → L1/Shared Memory → Registers → Matrix Unit
```

越靠近计算单元通常容量越小、延迟越低。只有数据确实会被重复使用时，提高 cache 命中才值得优先投入；Decode 的权重流通常复用有限，低 L2 hit 可能是工作负载属性。

### Memory Access：看每一级是否有效

不仅看 HBM，还要看 Global→L2、L2→Shared、Shared→Register 的访问方式；shared memory 还要检查 bank conflict。bank conflict 会把本应并行的访问串行化，即使 HBM、Tensor Core 和 occupancy 都没有明显异常。

### Pipeline：让数据供给和计算重叠

最简单的循环是 `load → wait → compute → wait`；double buffer 让当前 tile 计算时预取下一个 tile。pipeline 只能隐藏延迟，不能突破带宽上限，因此必须先知道瓶颈是什么。

## 5. 面向初学者的三层因果链

学习时始终把问题分成三层：

```text
第一层：问题本身
Shape / dtype / layout / FLOPs / Bytes

第二层：硬件映射
Tile / warp / block / Tensor Core / register / shared memory / occupancy

第三层：数据供给
HBM / L2 / shared access / bank conflict / prefetch / pipeline
```

最终链路是：

```text
HBM → L2 → Shared Memory → Registers → Tensor Core → Result
```

上游任意一级供数不足，Tensor Core 就会等待。看到 Tensor Core 利用率低时，先问“计算单元为什么没有持续得到数据”，不要直接把它当成计算指令问题。

## 6. 面向不同背景的学习方式

### 只会 Python/业务开发

先运行 `examples/gemm_learning_demo.py`，理解双重/三重循环、边界和 tile；再学习 C++ 指针、编译、GPU 内存和线程层级。不要一开始追求写出复杂 Kernel。

### 会 C/C++ 但没做 GPU

重点补 GPU 执行模型、warp/block、内存层次和同步；用同一个 GEMM 逐级改写，观察每次修改对应的指标变化。

### 会 CUDA 但没做性能分析

先练习从 profiling 反推根因，强制写下“观察—假设—修改—验证”；不要只看总耗时或单个 utilization。

### 做过 CUDA，准备迁移到其他加速器

把算子语义、CPU reference、测试、误差和 benchmark 放在公共层；把 kernel、compiler、runtime、同步和 profiler 放在后端层。不要写死 `warpSize==32`，线程组宽度和资源模型可能不同。

## 7. 阶段性毕业标准

完成入门阶段后，应该能独立回答：

1. 这个算子的输入输出和边界是什么？
2. 当前 shape 产生了多少并行工作？
3. 它更可能 Compute、Memory、Latency 还是 Launch Bound？依据是什么？
4. 线程访问地址是否合并，shared memory 是否有 bank conflict？
5. Occupancy 低的原因是什么，降低寄存器是否真的值得？
6. 修改后是算子变快了，还是只是某个指标变好看了？
7. 优化是否改善端到端性能，是否影响精度和其他 shape？

如果还不能回答这些问题，就继续做小实验；不要用更复杂的 Kernel 掩盖基础概念的不确定性。

## 本章参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
