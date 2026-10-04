# GEMM 完整案例：从六个数字到 tile、复用和性能证据

配套 [CPU 实验](../../examples/cpu/gemm_lab.py)、[朴素 CUDA](v1-naive/naive_gemm.cu) 与 [分块 CUDA](v2-tiled/tiled_gemm.cu)。本章不是“套一个高性能模板”，而是让你能解释每一次 load 为何发生、尾块如何正确、优化何时值得做。

## 1. 固定数学契约并手算

A 形状 M×K，B 形状 K×N，C 形状 M×N。取 A=[[1,2,3],[4,5,6]]、B=[[7,8],[9,10],[11,12]]。C[0,0]=1×7+2×9+3×11=58；C[0,1]=8+20+36=64；第二行是 [139,154]。因此 C=[[58,64],[139,154]]。这个例子 M=2、N=2、K=3，乘加数量 2MNK=24 FLOPs（按乘法和加法各算一次的常用估算）。

明确 A/B 是 row-major 还是 column-major、stride 是否可变、是否转置、累加 dtype、输出 dtype 和误差容差。若 row-major 连续，A[m,k] 地址是 baseA+m*K+k，B[k,n] 是 baseB+k*N+n，C[m,n] 是 baseC+m*N+n。B 的 K 维跨行，单线程逐 k 读 B 不等于访问连续；需要观察相邻线程 n 方向的合并访存。

## 2. V1：每线程算一个输出

网格按 M 和 N 做 ceil division。线程得到 (m,n)，若越界就退出；否则 acc=0，然后 k=0..K-1 累加 A[m,k]*B[k,n]，最后写 C[m,n]。这与最易理解的 CPU 三重循环一一对应。[V1 代码](v1-naive/naive_gemm.cu) 把它作为正确性基线，不能因为它慢就跳过；后续任何版本都要与同一 reference 比较。

同一行 C 的各线程会重复读 A，同一列 C 的线程会重复读 B。输出越多，重复访存越明显。朴素版也有 cache 命中，不能简单把源代码 load 次数等同于实际 HBM traffic；要用 profiler 检查。

## 3. V2：共享 tile 的计算账

设输出 tile 为 T×T。一个 block 负责 T² 个输出，每轮沿 K 取 A 的 T×T 片与 B 的 T×T 片进入 shared memory，然后每个线程用寄存器 acc 累加 T 个乘积。共有 ceil(K/T) 轮。对同一轮，A 的一个数可被 tile 内多个 n 复用，B 的一个数可被多个 m 复用。这就是减少全局内存重复取数的来源。

若 T=2，示例 K=3：第一轮处理 k=0,1，第二轮处理 k=2 和一个越界位置。越界 A/B 元素必须写 0 到 shared memory，而不是保留上一轮旧值。每轮“协作加载 → 同步 → 计算 → 同步”，第二次同步防止快线程覆盖仍被慢线程读取的 shared tile。输出尾块也只允许 m<M 且 n<N 的线程写回。对 M=37、N=29、K=41，三个维度都有尾块，现有 [V2 代码](v2-tiled/tiled_gemm.cu) 刻意测试了它。

T 增大可提升复用，却同时增加 shared memory、寄存器或线程需求，可能降低 occupancy。不要只凭 tile 大小判断快慢。

## 4. 在没有 GPU 时先验证算法

~~~powershell
python examples/cpu/gemm_lab.py
python -m unittest discover -s examples/cpu
~~~

CPU 实验打印同一矩阵的 naive 与 tiled 结果以及有效 tile load 计数。尝试 tile=1、2、4，以及 M/N/K 分别为 T-1、T、T+1 的组合；每次比较 reference。再制造一个 bug：把越界 tile 留作上轮内容，观察非整除 K 会出现哪类错误。CPU 实验只说明索引和语义，不代表 GPU shared memory、并行调度或速度。

有 CUDA 环境时按照根目录构建说明运行两个可执行文件；先看 correctness，再用固定机器、固定 shape、充分 warmup 计时。

## 5. 从时间到性能结论

GFLOPS=2MNK/(秒数×10^9)。例如相同 shape 的 1 ms 与 0.5 ms 分别对应两倍 GFLOPS，但若 shape 太小，计时受 launch 和测量噪声影响，不能推断算法上限。算术强度是 FLOPs/实际读写字节；理论最小流量不等于 profiler 观测流量，cache、重读、writeback 都会改变它。

至少测三类形状：大 M/N/K 的常规 GEMM；M=1 或很小 M 的 decode 型矩阵乘；三个维度都不整除 tile 的尾块。报告 kernel 时间、端到端时间、设备/驱动/编译器、dtype、layout 和误差。再用 profiler 检查是否是内存、计算、occupancy 或同步限制。V3 向量化必须先确认对齐和连续性；矩阵单元、异步搬运与流水线是后续硬件专精，不应伪装成此代码已有能力。

## 6. 练习与答案要点

练习 A：若 M=3、N=5、K=4、T=2，输出 block 网格为 ceil(3/2)×ceil(5/2)=2×3，K 轮数 2。练习 B：若 K=5、T=2，最后一轮第 2 个 k 槽为什么置零？它越界，置零才能保证点积不加入无效旧数据。练习 C：M=1 时为什么大方阵 GEMM 的最优配置未必适用？输出并行度和权重复用结构不同，需要单独 profile。

## 7. 四个线程的装载表：tile 究竟长什么样

继续使用 A 为 2×3、B 为 3×2 的例子，设 TILE=2。block 中四个线程按 (ty,tx) 编号，负责的输出为 C[ty,tx]。第一轮 k0=0：

| 线程 | 装载到 As[ty,tx] | 装载到 Bs[ty,tx] | 最终负责的输出 |
|---|---|---|---|
| (0,0) | A[0,0]=1 | B[0,0]=7 | C[0,0] |
| (0,1) | A[0,1]=2 | B[0,1]=8 | C[0,1] |
| (1,0) | A[1,0]=4 | B[1,0]=9 | C[1,0] |
| (1,1) | A[1,1]=5 | B[1,1]=10 | C[1,1] |

装载后的两个共享矩阵是：

~~~text
As = [[1,2],     Bs = [[7, 8],
      [4,5]]           [9,10]]
~~~

线程 (0,0) 自己只搬了 A 的 1 和 B 的 7，但它最终还需要线程 (0,1) 搬来的 2、线程 (1,0) 搬来的 9。shared memory 和加载后的屏障正是为这类跨线程使用服务。

第一轮完成后四个寄存器累加器为：

~~~text
acc00 = 1*7 + 2*9  = 25
acc01 = 1*8 + 2*10 = 28
acc10 = 4*7 + 5*9  = 73
acc11 = 4*8 + 5*10 = 82
~~~

第二轮 k0=2，K 的最后一个有效位置为 2，位置 3 必须补零：

~~~text
As = [[3,0],     Bs = [[11,12],
      [6,0]]           [ 0, 0]]
~~~

这轮增量是 [[33,36],[66,72]]。加到旧累加器，得到 [[58,64],[139,154]]。累加器必须在 K 循环外初始化为 0；如果每轮重置，就只剩最后一轮增量。

运行 [数据流脚本](../../examples/cpu/dataflow_lab.py) 会逐轮打印两个共享 tile 与累加器：

~~~powershell
python examples/cpu/dataflow_lab.py --topic gemm
~~~

这些状态对应现有 CUDA V2 的加载、计算循环和 acc 变量，可以逐项对照，而不必从整段 kernel 猜执行过程。

## 8. 地址公式的四个坐标，不要混在一起

在 [tiled_gemm.cu](v2-tiled/tiled_gemm.cu) 中：

~~~cpp
row   = blockIdx.y * TILE + threadIdx.y;
col   = blockIdx.x * TILE + threadIdx.x;
a_col = tile * TILE + threadIdx.x;
b_row = tile * TILE + threadIdx.y;
~~~

row/col 是输出坐标，贯穿所有 K 轮保持不变；a_col/b_row 是本轮装载的 K 坐标，会随 tile 变化。A 的索引为 row*K+a_col，B 的索引为 b_row*N+col。把 b_row 错写成 row，可能在方阵/特殊输入上碰巧通过，但在 M≠K 时暴露。

共享内存的坐标又是局部坐标：As[ty][tx] 和 Bs[ty][tx]。进入内层乘加，取 As[ty][inner] 与 Bs[inner][tx]；它们分别沿 As 的行和 Bs 的列读取。不能把“装载时用 tx/ty”与“计算时用 inner”混为一谈。

## 9. 用一个非整除输出块证明边界策略

取 M=3、N=3、K=3、TILE=2，右下输出块基址为 (2,2)，四个线程的 row/col 是 (2,2)、(2,3)、(3,2)、(3,3)。只有第一个线程输出合法，但其余线程仍有协作任务。

例如线程 (0,1) 的输出 col=3 越界，但它在第一轮负责把 A[2,1] 放入共享内存；合法输出 C[2,2] 正需要这个数。若看到输出越界就提前 return，既会导致某些装载缺失，也会让 block 内屏障无法按要求执行。

所以 V2 的策略是：所有线程参与加载和同步；每次 A/B load 分别判断自己的边界，无效值写 0；只在最终 store 时判断输出合法性。不是所有越界都由同一个 if(row<M && col<N) 处理。

## 10. 两次屏障分别保护哪个时间段

第一个屏障保护“生产者写 tile → 消费者读 tile”。如果缺少它，线程可能读到尚未写入的元素。第二个保护“消费者读旧 tile → 下一轮生产者覆盖 tile”。如果缺少它，快线程进入下一轮后可能用新值覆盖慢线程仍需的旧值。

把这段生命周期画成时间线：

~~~text
加载 round 0 → barrier → 使用 round 0 → barrier
加载 round 1 → barrier → 使用 round 1 → barrier
~~~

最后一轮后的第二个屏障理论上可以在严谨证明不再复用共享存储后省略，但教学 baseline 保持统一结构更易审查。性能优化时应通过工具和实测判断屏障是否值得专门处理。

## 11. 分块究竟少读了多少：先写清计算模型

忽略尾块和 cache，每个 T×T 输出块有 T² 个输出，每个输出需要 K 对输入。朴素源码层面的输入读取为 2T²K 个标量。共享分块每轮读 2T² 个标量，共 K/T 轮，即 2TK 个标量。理想模型中的读取次数比为 T。

这个结论的前提是“比较 block 内重复 load 请求”。它不等于真实 HBM 字节必然缩小 T 倍，因为朴素版也可能从 cache 命中，tile 版也有尾块/资源开销。

每一轮做 2T³ FLOPs，加载 2T² 个 FP32 数，即 8T² 字节。因此只计这轮输入的算术强度约 T/4 FLOPs/byte。T=16 时为 4，T=32 时为 8。但还要考虑输出写回、实际 K 长度、cache 和设备执行路径，不能直接由这个数预测 GFLOPS。

## 12. 为什么一个线程算多个输出可能更快

目前每线程只有一个 acc。寄存器分块让一个线程维护一个小输出片，例如两个行方向的累加器 acc0/acc1。当线程读到同一个 B 值时，可以与两个 A 值分别相乘；B 在该线程寄存器中复用，两条乘加也可能提供更多独立指令。

代价是多了 accumulator、索引和临时变量。寄存器资源由整个 SM 上的驻留线程共同占用，单线程寄存器增多可能使驻留 block 数下降，甚至发生 spill。spill 是寄存器中的值转移到 local memory 访问，不能把名称中的 local 误解为快速的 block shared memory。

因此“寄存器分块更先进”不等于每个尺寸都更快。先测小输出 tile 与大输出 tile，再看寄存器报告、local memory 访问和绝对时间。

## 13. 向量化加载前必须算地址对齐

float4 代表一次访问四个连续 FP32 值，需要满足该访问的自然对齐和有效范围。即使分配返回的 base 对齐，第二行起点未必对齐。例如 row-major A 的 K=41，每行 164 字节；第二行比第一行偏移 164 mod 16=4 字节。直接把每一行起点 reinterpret_cast 为 float4 指针就不成立。

即使行起点对齐，剩下 1～3 个元素时也不能四个一起越界读。应为完整且对齐的片段使用向量路径，尾部走标量路径，或者定义并验证更严格的输入条件。

合并访存与向量化不同：前者看同一 warp 的多个线程访问怎样组合，后者看单线程指令怎样读取多个元素。标量 load 可以合并得很好；向量 load 也可能因为跨行/跨度问题浪费流量。详见 [V3 对齐实验](v3-vectorized/README.md)。

## 14. Tensor Core 与流水线到底改变了哪一层

矩阵指令让线程组共同计算小矩阵片，不再是“每个线程独立做一个标量乘加”这么简单。需要明确输入片的布局、支持 dtype、累加 dtype、tile 形状与架构条件。编译目标不支持某项指令时，模板中出现相关名字并不能证明硬件实际执行它。

双缓冲/流水线把“下一块数据搬运”和“当前块计算”重叠，理想上减少等待。但它需要额外缓冲空间和严格的生产/消费顺序。如果计算太短、数据本就在 cache、或者额外资源导致并发下降，流水线的收益可能很小甚至为负。

本仓库 V2 仍是同步共享内存教学实现；阅读 CUTLASS 时，用这两层变化去区分“计算指令升级”与“数据供应重叠”，不要把所有技巧混成一个黑盒优化。

## 15. 一个能证伪优化假设的实验设计

假设：“T 从 16 改到 32，复用提高，因此更快。”先设置三类输入：大方阵、M=1 的瘦矩阵、M/N/K 都有尾块的矩阵。分别记录正确性、kernel 时间、总时间、寄存器/shared memory 占用与活跃 block 情况。

如果大方阵变快而 M=1 变慢，说明收益依赖输出形状。M=1 时大的二维线程块有大量无效输出线程，而且跨行复用机会消失；需要形状特化调度。若所有尺寸都变慢，检查 T=32 带来的 1024 线程 block 是否限制驻留或资源，而不应立即宣称 shared memory 没用。

练习：把当前源码 TILE 从 16 改为 32 之前，先写下会同时变化的线程数、两个共享 tile 的字节数和 K 轮数。答案：线程数由 256 变为 1024，共享 A/B 从 2×16²×4=2048 字节变为 8192 字节，K 轮数从 ceil(K/16) 变为 ceil(K/32)。这是三个联动变量，解释结果时都要考虑。

## 参考资料（按学习顺序）

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：block、shared memory、同步。
2. [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：理解形状和硬件利用率。
3. [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：对照 tile、mask 与 autotuning。
4. [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)：进阶分层 GEMM 设计。
