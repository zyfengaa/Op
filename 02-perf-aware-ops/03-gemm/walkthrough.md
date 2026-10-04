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

## 参考资料（按学习顺序）

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：block、shared memory、同步。
2. [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：理解形状和硬件利用率。
3. [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：对照 tile、mask 与 autotuning。
4. [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)：进阶分层 GEMM 设计。
