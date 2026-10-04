# Reduce 的一次完整开发经历

任务：把一个向量归约成标量。沿着串行、block 协作、跨 block 合并三步，观察计算、同步、launch 和临时存储的变化。先完成 [树形归约手算](walkthrough.md)。

## ① 固定 reference 和算子契约

本例 Sum 接受连续 FP32 输入，输出一个值，空输入在 host 返回 0。Max/Mean 的空输入策略不能直接沿用 Sum。本例的四分之一输入方便检查索引；一般浮点数据还需高精度 reference 与容差。

[operator_ladders.cu](../../examples/cuda/operator_ladders.cu) 的 sum_v0 由一个线程串行累加，易于理解但大输入并行度低。

## ② 从一个 block 到完整两阶段归约

| 版本 | 算法 | 设备启动次数 | 主要代价 |
|---|---|---:|---|
| sum_v0 | 单线程扫描 | 1 | 缺少并行工作 |
| V1 | 每线程局部和 + shared tree，一 block | 1 | 单个 block 限制设备利用 |
| V2 | 多 block 产生 partial，再归约 partial | 2 | 临时数组、第二次 launch |

V1/V2 复用 sum_blocks，但 grid 不同；V2 的第二次输入是 partial 数组，不再是原向量。两个 kernel 在同一 stream 顺序执行，提供生产者/消费者顺序。每个 block 的所有线程都参加屏障，无输入的线程贡献 0。

### 追踪一次 1 到 10 的归约

手算时取 4 个线程，实际源码取 256 个。V1 的线程局部和是 t0=1+5+9=15、t1=2+6+10=18、t2=3+7=10、t3=4+8=12。shared 最初为 [15,18,10,12]；stride=2 后有效前半为 [25,30]；stride=1 得到 55。屏障保证下一轮读到上一轮已经完成的结果。

若第一阶段用 2 个 block，grid stride=8。block0 的四个线程分别得到 [10,12,3,4]，归约为 29；block1 得到 [5,6,7,8]，归约为 26。第二个 kernel 的输入只是 [29,26]，结果仍为 55。这个例子刻意把“线程局部累加”“block 内合并”“跨 block 合并”分开，避免把它们当成同一个同步范围。

临时 partial 占 block_count×sizeof(float)，不是 N×sizeof(float)。它很小也仍有写入、读取和第二次启动成本。若只比较第一阶段用时，会让 V2 看起来比实际完整算子更有优势。

~~~mermaid
flowchart LR
  X[原输入 N 项] --> G[多个 block 各做局部归约]
  G --> P[partial：每 block 一个结果]
  P --> K[同一 stream 中的第二个 kernel]
  K --> Y[一个输出]
~~~

这次已提供实际两阶段 CUDA 代码。它是性能候选，不预先命名为最快版；小 N 时第二次 launch 可能抵消并行收益。

~~~powershell
cmake --build build/cuda --config Release
.\build\cuda\Release\operator_ladders.exe --benchmark
~~~

先按根目录说明配置 CMake。程序逐版本验证后才计时，V2 的事件范围包含两次 kernel；只测第一阶段会漏掉完成算子的成本。当前本机没有 CUDA 工具链，这些设备结果仍待验证。

## ③ 四个归约故障

~~~powershell
python exercises/check.py show RD01
python exercises/check.py grade RD01
python exercises/check.py grade RD02
python exercises/check.py grade RD03
python exercises/check.py grade RD04
~~~

题目依次暴露：线程只处理第一段、全负 Max 被 padding 污染、Mean 用错分母、非 2 的幂树漏项。每次先推断哪一类 shape 会通过，再看回归输入。

RD04 的修复可以补齐到 2 的幂，也可以设计正确的奇数长度合并；判题只约束语义。真正设备实现还需评价额外工作、内存和分支。

## ④ 一份可以对照的 profiler 证据

[CPU 基线报告](../../labs/results/cpu-reference/README.md) 包含 Python 顺序累加、Python 树、PyTorch 原生 sum 的真实样本和 trace。Python 树在 CPU 解释器上引入列表分配，并不会因为数学上有 logN 层就自动获得并行加速。

原生报告中 aten::sum 为 65536 个 float64 输入产生 8 字节输出。关注输入/输出规模、调用层级和计时范围；CPU Mem 是事件的净分配信息，不是整个算子的峰值内存。

CUDA 预期：V0 缺少并行性；V1 只有一个 block；V2 有多个第一阶段 block 和一个第二阶段 block。用 Nsight 时间线确认这两个阶段，再看每段 duration 与整体事件时间。没有 ncu 实测时，不给寄存器数、带宽或 occupancy 编造标准值。

## ⑤ 平台迁移实验

[第二后端实操](../../05-platform-specialization/second-backend.md) 复用相同输入，对比 Python 与 PyTorch CPU。迁移到另一个设备时，Sum 的语义仍在，但归约树、组宽、同步和累加精度需要重做。不要把“同组线程在一起执行”当成自动同步。

Max 的中性值、Mean 的 count、浮点确定性与容差属于公共契约；warp 宽度、shuffle mask 和 shared memory 容量属于平台实现。

## 与模型主线的连接

RMSNorm 用平方和/均值，Softmax 用 max 与指数和。进入 [完整 block](../../06-model-slice/README.md)，观察两个 RMSNorm 和 Attention，解释为何一个“归约方法”会出现在多个模型阶段，且归约轴不一定相同。

## 交付判定

提交 V0/V1/V2 的调用链和完整时间范围、RD01–RD04 修复、一份实际报告、一个加法顺序导致误差的例子，以及“哪些语义跨后端不变”的说明。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：block 同步与执行顺序。
- [CUB DeviceReduce](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)：生产库的设备级归约接口。
- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：稳定测量的基本方法。
