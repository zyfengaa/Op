# 算子开发入门：从第一个 CUDA Kernel 到性能分析

这是一份给没有做过算子开发的开发人员使用的主教材。请按顺序阅读：先理解数据和 CPU 程序，再理解 GPU 线程，最后学习 Tensor Core 和性能优化。

文中的性能数字是教学示例；真实性能必须在目标设备上重新测量。

---

## 第 0 章：算子开发到底在做什么

### 0.1 算子是什么

在深度学习框架中，add、relu、matmul、softmax、layer_norm 都可以看作算子。一个算子描述的是：给定输入张量，按照明确的数学规则产生输出张量。

Vector Add：

~~~text
C[i] = A[i] + B[i]
~~~

矩阵乘法：

~~~text
C[m,n] = sum(A[m,k] * B[k,n])
~~~

完整的算子开发包括：

1. 定义数学语义和输入输出契约。
2. 写可信的 CPU reference。
3. 设计并行算法和内存访问方式。
4. 实现 GPU kernel。
5. 证明结果正确。
6. 测量性能并定位瓶颈。
7. 集成到 PyTorch、推理框架或其他 runtime。
8. 对不同 shape、dtype、stride 和设备做回归。

### 0.2 新人最容易犯的错误

代码越复杂不代表越快。复杂代码可能增加寄存器、同步和维护成本。

Tensor Core 利用率低也不代表 Tensor Core 有问题。如果工作负载是 Memory Bound，计算单元可能只是在等待数据。

程序能运行也不代表算子正确。只测试整除 tile 的尺寸，可能完全漏掉尾块越界和错误 stride。

### 0.3 每个实验的固定模板

~~~text
问题：我要计算什么？
输入：shape / dtype / layout / stride
Reference：CPU 如何得到正确答案？
实现：线程和内存如何映射？
正确性：和 reference 的误差是多少？
Baseline：当前耗时、带宽或 GFLOPS 是多少？
假设：哪个指标说明了什么问题？
修改：本次只改一个主要变量。
验证：指标和总时间是否按预期变化？
结论：假设成立、部分成立，还是被证伪？
~~~

### 本章练习

1. 为什么需要 CPU reference？
2. 为什么必须记录 shape 和 dtype？
3. 为什么平均耗时比最好一次耗时更可信？

### 本章参考资料

- CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA C++ Best Practices Guide：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

---

## 第 1 章：C/C++、矩阵和编译基础

### 1.1 二维矩阵其实是一维内存

row-major 矩阵 A[rows][cols] 的地址公式是：

~~~text
address(A[row][col]) = base + (row * stride + col) * sizeof(element)
~~~

连续矩阵的 stride 通常等于 cols，但切片矩阵的 stride 可能更大。生产算子必须区分逻辑尺寸和物理 stride。

~~~cpp
std::vector<float> a(rows * cols);
for (int row = 0; row < rows; ++row) {
  for (int col = 0; col < cols; ++col) {
    a[row * cols + col] = float(row * cols + col);
  }
}
~~~

### 1.2 编译最小 C++ 程序

~~~cpp
#include <iostream>
int main() {
  std::cout << "hello operator" << std::endl;
  return 0;
}
~~~

Windows：

~~~powershell
cl /std:c++17 hello.cpp
~~~

Linux：

~~~bash
g++ -std=c++17 -O2 hello.cpp -o hello
~~~

CUDA 文件：

~~~bash
nvcc -std=c++17 -O2 -lineinfo vector_add.cu -o vector_add
~~~

lineinfo 让 profiler 更容易关联源代码行，它不是性能优化开关。

### 1.3 浮点误差

并行归约改变了浮点加法顺序，因此不能用 actual == expected。常用判断是：

~~~text
abs(actual - expected) <= atol + rtol * abs(expected)
~~~

### 本章练习

1. 写 CPU 矩阵加法，支持任意 rows 和 cols。
2. 打印 A[2][3] 对应的一维下标。
3. 故意使用错误 stride，观察后续行为什么错误。

### 本章参考资料

- C++ vector：https://en.cppreference.com/w/cpp/container/vector
- CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Best Practices 数值精度：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

---

## 第 2 章：GPU 执行模型

### 2.1 Thread、Warp、Block、Grid

CUDA 的层级关系是：

~~~text
Grid
 └── Block
      └── Warp
           └── Thread
~~~

一个 block 会被调度到一个 SM 上。block 内线程可以使用 shared memory 和 __syncthreads。普通 kernel 中不能把不同 block 当成已经同步。

### 2.2 一维索引

~~~cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
~~~

假设 blockDim.x 为 256：

~~~text
block 0, thread 0   -> i=0
block 0, thread 255 -> i=255
block 1, thread 0   -> i=256
~~~

### 2.3 二维索引

~~~cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
  c[row * N + col] = a[row * N + col] + b[row * N + col];
}
~~~

x 映射列、y 映射行是常见约定，但不是硬件强制要求。关键是索引、布局和地址计算必须一致。

### 2.4 为什么必须边界判断

N=1000、block=256 时：

~~~text
grid = ceil(1000 / 256) = 4
总线程 = 1024
~~~

最后 24 个线程没有有效数据。没有边界判断就可能越界读写。

### 本章练习

1. 画出 N=10、block=4 的线程对应关系。
2. 把 Vector Add 改成二维矩阵加法。
3. 删除边界判断，观察越界错误。

### 本章参考资料

- CUDA Programming Model：https://docs.nvidia.com/cuda/cuda-programming-guide/
- Writing SIMT Kernels：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html

---

## 第 3 章：第一个算子 Vector Add

### 3.1 CPU reference

~~~cpp
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
~~~

它不追求快，只负责定义正确答案。

### 3.2 CUDA kernel

~~~cpp
__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
~~~

逐行理解：

- __global__：CPU launch，GPU 执行。
- i：当前线程负责的元素编号。
- if：处理最后一个不完整 block。
- c[i]：一个线程只写自己的输出位置。

### 3.3 Launch 和错误检查

~~~cpp
int block = 256;
int grid = (n + block - 1) / block;
vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
~~~

cudaGetLastError 检查 launch 和异步错误；cudaDeviceSynchronize 等待 kernel 完成并把错误报告到 CPU。

### 3.4 测试矩阵

至少测试 0、1、31、32、33、255、256、257、100003 个元素。重点是不能整除 block 的尺寸。

### 3.5 性能理解

Vector Add 每个元素只做一次加法，却要读取 A、B 并写 C，通常是 Memory Bound：

~~~text
理论 bytes ≈ 3 * N * sizeof(float)
带宽 ≈ bytes / kernel_time
~~~

不要用 Vector Add 的 GFLOPS 和 GEMM 比较，因为算术强度不同。

### 本章练习

1. 增加 ReLU：y[i] = max(x[i], 0)。
2. 增加 scale：y[i] = alpha * x[i] + beta。
3. 对比融合和拆成两个 kernel 的中间流量。

### 本章参考资料

- CUDA Vector Add 示例：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Global Memory：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-accesses

---

## 第 4 章：Layout、Stride 和合并访存

### 4.1 连续访问为什么重要

GPU 以 warp 为单位执行。如果一个 warp 的线程访问相邻地址，硬件通常可以用较少的 transaction 服务它们；如果线程访问跨度很大，就会产生更多 transaction。

row-major 地址：

~~~text
A[row][col] = base + (row * stride + col) * sizeof(dtype)
~~~

让相邻线程变化 col，通常是连续访问；让相邻线程变化 row，跨度通常为一个 stride。

### 4.2 两种线程映射

较好的映射：

~~~text
thread 0 -> A[row][0]
thread 1 -> A[row][1]
thread 2 -> A[row][2]
~~~

可能较差的映射：

~~~text
thread 0 -> A[0][col]
thread 1 -> A[1][col]
thread 2 -> A[2][col]
~~~

第二种相邻地址可能相差 stride * sizeof(dtype)。最终应使用 profiler 验证，而不是只凭图猜测。

### 4.3 stride 不能默认等于 width

一个矩阵可能是大矩阵的切片，逻辑宽度为 100，物理 stride 为 128。kernel 如果使用 100 作为 stride，前几行可能看似正常，后面会读取错误位置。

接口应明确传递 rows、cols 和 stride。

### 本章实验

固定矩阵大小，只改变线程映射，记录：

~~~text
Kernel time
Requested bytes
Actual memory traffic
Global Load Efficiency
L2 transactions
~~~

显存吞吐高不等于有效带宽高；actual traffic 大于 requested bytes 说明事务存在浪费。

### 本章参考资料

- CUDA Coalesced Access：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
- CUDA Memory Hierarchy：https://docs.nvidia.com/cuda/cuda-programming-guide/

---

## 第 5 章：Reduce 与 Softmax

### 5.1 Reduce 为什么困难

Vector Add 的每个输出只依赖一个输入；Reduce 的一个输出依赖很多输入：

~~~text
y = x[0] + x[1] + ... + x[N-1]
~~~

线程之间需要共享部分结果、同步，并处理浮点加法顺序。

### 5.2 Block Reduce

每个线程先累加一部分：

~~~text
partial[thread] = x[thread] + x[thread + blockDim.x] + ...
~~~

然后写入 shared memory，使用 tree reduction：

~~~text
stride=block/2：前半线程加后半线程
stride=block/4：再合并一半
...
stride=1：得到 block sum
~~~

每轮访问 shared memory 后都要在需要的地方同步。

### 5.3 Multi-block Reduce

当 N 很大，一个 block 不够：

1. 第一个 kernel 产生每个 block 的 partial sum。
2. 第二个 kernel 对 partial sums 再归约。

普通 kernel 中不能假设所有 block 会同时到达某个全局同步点。

### 5.4 Stable Softmax

直接 exp(x) 可能溢出。稳定公式：

~~~text
m = max(x)
e[i] = exp(x[i] - m)
s = sum(e[i])
y[i] = e[i] / s
~~~

Row-wise softmax 通常需要求 max、求 sum、写结果。fused 版本尽量把中间值留在寄存器或 shared memory，减少 Global Memory 往返。

### 5.5 测试

测试普通值、大正数、大负数、相同最大值、很长 row 和非 32 对齐长度。验证每行输出和接近 1，并与高精度 reference 比较。

### 本章参考资料

- CUDA Synchronization：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Best Practices：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- Triton Fused Softmax：https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

---

## 第 6 章：Transpose 和 Shared Memory

### 6.1 Naive Transpose

~~~text
Y[col,row] = X[row,col]
~~~

简单 kernel 可能连续读取 X，却跨行写 Y，读写一侧很难同时合并。

### 6.2 Tiled Transpose

把输入 tile 合并读入 shared memory，再以转置方向写出：

~~~text
Global X -> Shared Tile -> Global Y
~~~

shared memory 是 block 内线程共享的 scratchpad。写入完成后要同步，避免读取未完成的数据。

### 6.3 Bank Conflict

shared memory 被划分成多个 bank。一个 warp 的多个线程访问同一个 bank 时，访问可能串行化。常见教学方案是把 [TILE][TILE] 改成 [TILE][TILE+1]，但是否有收益必须结合实际访问模式和指标验证。

### 6.4 验证

测试矩形矩阵、方阵、1×N、N×1 和非 tile 整除尺寸。若输入 rows×cols，输出必须是 cols×rows。有效带宽包括一次读和一次写：

~~~text
bytes = 2 * rows * cols * sizeof(dtype)
~~~

### 本章参考资料

- CUDA Shared Memory：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Shared Memory Best Practices：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory

---

## 第 7 章：从 CPU GEMM 到 naive CUDA GEMM

### 7.1 M/N/K

~~~text
A[M,K] * B[K,N] = C[M,N]
~~~

M 是输出行数，N 是输出列数，K 是点积长度。总 FLOPs 约为 2*M*N*K。它们还决定并行度、tile 尾块和矩阵指令适配。

### 7.2 CPU reference

~~~cpp
for (int m = 0; m < M; ++m) {
  for (int n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
  }
}
~~~

### 7.3 Naive CUDA

一个线程负责一个 C 元素：

~~~cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}
~~~

它正确但会重复加载 A/B。不同线程计算相邻 C 时，很多输入本可以复用，却被重复从 Global Memory 取出。

### 7.4 Benchmark

~~~text
GFLOPS = 2 * M * N * K / seconds / 1e9
~~~

kernel 计时使用 CUDA Event；不要把一次性分配和拷贝混入 kernel benchmark。端到端测试则单独报告 copy、kernel 和 total。

### 本章参考资料

- CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Performance Metrics：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUTLASS：https://docs.nvidia.com/cutlass/

---

## 第 8 章：GEMM Tiling 和 Shared Memory

### 8.1 为什么 Tiling 有效

对于输出 tile BM×BN，A tile 为 BM×BK，B tile 为 BK×BN。把它们搬到 shared memory 后，多个线程重复使用同一块数据：

~~~text
load A tile, load B tile
sync
compute C tile
sync
load next K tile
~~~

### 8.2 ceil grid 和尾块

~~~text
grid_m = (M + BM - 1) / BM
grid_n = (N + BN - 1) / BN
~~~

读取 A/B 时，越界位置填 0；写回 C 时只有 row<M 且 col<N 才能写。必须测试 M=4103、N=4099、K=4097。

### 8.3 tile 不是越大越好

更大的 tile 可能增加复用，却也增加 shared memory、线程数、寄存器压力和尾块浪费，并可能降低 occupancy。因此要对多个 tile 尺寸实测。

### 8.4 验证假设

如果 tiling 后时间下降，继续确认 Global Memory traffic 是否下降、shared memory 是否有 bank conflict、occupancy 是否因为资源增加而下降。不要只记录最终耗时。

### 本章参考资料

- CUDA Shared Memory：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory
- CUDA Coalesced Access：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
- CUTLASS Efficient GEMM：https://docs.nvidia.com/cutlass/

---

## 第 9 章：Register Tiling、Warp Mapping 和 Occupancy

### 9.1 一个线程计算多个结果

让线程保留小块 accumulator，可以提升寄存器内数据复用，减少 shared memory 往返。但 accumulator 会消耗寄存器。

### 9.2 Occupancy

粗略理解：

~~~text
occupancy = active warps / hardware maximum warps
~~~

限制因素包括 registers/thread、shared memory/block、threads/block 和硬件上限。

假设一个 SM 有 65536 个寄存器，一个 block 有 256 线程，每线程 128 个寄存器：

~~~text
一个 block = 256 * 128 = 32768 registers
最多约 2 个 block 常驻
~~~

### 9.3 Occupancy 不是越高越好

Kernel A occupancy 75%、Tensor Core 55%；Kernel B occupancy 50%、Tensor Core 85%，B 可能更快，因为 B 具有更好的数据复用和指令级并行。目标是足以隐藏延迟并持续供给计算单元。

### 9.4 实验

只改变每线程 accumulator 数量，记录 registers/thread、occupancy、Long Scoreboard Stall、Tensor Core active 和 kernel time。如果 occupancy 上升而时间变慢，说明可能损失了数据复用。

### 本章参考资料

- CUDA Hardware Implementation：https://docs.nvidia.com/cuda/cuda-programming-guide/
- Nsight Compute：https://docs.nvidia.com/nsight-compute/
- CUDA Best Practices：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

---

## 第 10 章：Tensor Core 和 MMA

### 10.1 Tensor Core 做什么

Tensor Core 不是一个线程计算一个结果，而是 warp 或 warpgroup 协同执行固定形状的矩阵乘加指令：

~~~text
fragment A * fragment B + accumulator C
~~~

### 10.2 FP16 不等于 Tensor Core

即使输入是 FP16，也可能因为以下原因没有走目标路径：

- shape 或 K 不满足要求。
- layout 不符合 fragment 装载方式。
- 编译器选择了普通 CUDA Core。
- warp 大部分时间在等待数据。
- 尾块走了 fallback。

必须检查 dtype、layout、编译产物和 profiler 指标。

### 10.3 规整和非规整 shape

4096×4096×4096 容易切成完整 tile；4103×4099×4097 会出现 partial tile，需要 mask、padding 或 fallback。规整 shape 上利用率高，不能证明任意 shape 都高。

### 10.4 性能判断

高 AI 的大 GEMM 更可能 Compute Bound，应关注 MMA 发射和 Tensor Core throughput；小 M Decode GEMM 可能 AI 很低，首要问题是权重流和 latency hiding，而不是提高 Tensor Core utilization。

### 本章参考资料

- CUDA Tensor Cores：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUTLASS：https://docs.nvidia.com/cutlass/
- NVIDIA Matrix Multiplication Background：https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

---

## 第 11 章：Pipeline、Double Buffer 和 Async Copy

### 11.1 没有重叠时的时间线

~~~text
LOAD0 -> WAIT -> COMPUTE0 -> WAIT -> LOAD1 -> WAIT -> COMPUTE1
~~~

double buffer 用两块 buffer，让当前 tile 计算时预取下一 tile：

~~~text
LOAD0
       COMPUTE0 + LOAD1
                    COMPUTE1 + LOAD2
~~~

### 11.2 正确性风险

必须证明：

1. compute 读取的 buffer 已经完成 load。
2. load 不会覆盖仍在使用的 buffer。
3. 线程在切换 buffer 前处于一致状态。
4. 尾部不会访问不存在的下一 tile。

### 11.3 何时没有收益

如果 HBM 已接近饱和，pipeline 只能隐藏延迟，不能突破带宽上限。如果 kernel 受 launch overhead 或并行度不足限制，double buffer 可能只增加资源使用。

### 11.4 验证

观察 load/compute 时间线、Long Scoreboard、Tensor Core active 和实际总时间。如果指标没有按假设变化，就回退修改并重新分析。

### 本章参考资料

- CUDA Asynchronous SIMT：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Async Copy：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUTLASS Pipeline Concepts：https://docs.nvidia.com/cutlass/

---

## 第 12 章：Roofline、Benchmark 和 Nsight

### 12.1 Arithmetic Intensity

~~~text
AI = FLOPs / transferred bytes
Roofline knee = peak compute / peak bandwidth
~~~

AI 远低于 knee，先怀疑 Memory Bound；AI 很高，才重点看 Compute Bound。实际 traffic 可能大于理论值，因此 Roofline 是初始分类，不是最终答案。

### 12.2 工具分工

Nsight Systems 回答“哪个阶段或哪个 Kernel 占总时间”；Nsight Compute 回答“这个 Kernel 为什么慢”。

如果 Top Kernel 不清楚，先用 Systems 看全局时间线，再用 Compute 做细粒度分析。

### 12.3 诊断流程

~~~text
Top Kernel
-> 确认模型层
-> 确认 M/N/K、dtype、layout
-> 计算 FLOPs 和理论 bytes
-> Roofline 分类
-> 检查 bandwidth / Tensor Core / occupancy
-> 检查 global/shared access 和 bank conflict
-> 检查 registers、stall、pipeline
-> 只改一个变量
-> 重新 profiling
~~~

### 12.4 Benchmark 规范

- 固定 GPU、驱动、编译选项和输入。
- 预热，排除首次加载影响。
- 固定重复次数，报告均值、P50/P90 或标准差。
- 分离 Host-device copy、kernel 和 E2E。
- 保存命令、设备信息、报告文件和源码版本。

### 本章参考资料

- Nsight Compute User Guide：https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- Nsight Compute Profiling Guide：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Compute Triage Guide：https://docs.nvidia.com/nsight-compute/ComputeTriage/
- Nsight Systems：https://docs.nvidia.com/nsight-systems/

---

## 第 13 章：PyTorch、Triton 和框架集成

### 13.1 为什么集成不是包一层 Python

框架调用会带来 dtype、device、stride、autograd、dispatch、stream 和错误处理问题。独立 kernel 正确不等于框架算子正确。

### 13.2 PyTorch 集成步骤

1. 定义 Python API 和算子 schema。
2. C++ 层检查 device、dtype、shape 和 stride。
3. CUDA 层只接收已验证参数。
4. 注册 CPU/CUDA dispatch。
5. 用 reference、随机测试、边界测试和 opcheck 验证。
6. 再做 micro benchmark 和模型 E2E benchmark。

### 13.3 Triton 的定位

Triton 用 Python 风格描述 block/program mapping，适合快速实验 Vector Add、Softmax、Matmul 和 LayerNorm。它降低实现门槛，但不会自动替你决定 shape、layout、tile 和性能根因。

### 13.4 集成测试清单

~~~text
CPU correctness
CUDA correctness
不同 dtype
不同 device
contiguous / non-contiguous
空输入和小输入
非整除 shape
autograd（若支持训练）
opcheck / stream
~~~

### 本章参考资料

- PyTorch Custom C++ and CUDA Operators：https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- PyTorch Extension Overview：https://docs.pytorch.org/tutorials/extension.html
- Triton Tutorials：https://triton-lang.org/main/getting-started/tutorials/

---

## 第 14 章：完整项目实战和跨平台迁移

### 14.1 项目题目

实现 fused GEMM：

~~~text
Y = SiLU(A * B + bias)
~~~

要求保留 CPU reference、naive CUDA、tiled CUDA 和 fused 版本。

### 14.2 项目目录

~~~text
operator_project/
├── CMakeLists.txt
├── include/              公共接口和 shape 检查
├── src/reference/        CPU reference
├── src/cuda/             CUDA kernels
├── tests/                correctness 和边界测试
├── benchmarks/           micro benchmark
├── profiles/             Nsight 报告
└── docs/                 设计与性能报告
~~~

### 14.3 交付顺序

1. 定义公式、shape、dtype、layout、stride 和误差标准。
2. 写 reference 和固定测试向量。
3. 建立框架 baseline。
4. 写 naive kernel，确保结果正确。
5. 加入 tiling，一次只改一个变量。
6. 用 profiler 证明瓶颈和收益。
7. 做 fusion，比较中间结果读写减少了多少。
8. 加入框架集成、回归测试和性能门槛。
9. 写限制条件：支持的 shape、dtype、设备和退化场景。

### 14.4 跨平台边界

可迁移：

- 数学语义。
- shape、stride、dtype 契约。
- CPU reference。
- 正确性测试和误差标准。
- benchmark 方法。
- 性能假设和证据链。

通常需要重写：

- kernel 语法。
- 编译器和 runtime。
- 线程组模型。
- 同步和异步拷贝 API。
- profiler 指标。
- 矩阵专用指令和高性能库。

不要把 warpSize 等于 32、某个 CUDA intrinsic 或某个 Tensor Core tile 当作跨平台规则。

### 14.5 毕业标准

拿到一个慢 Kernel 后，应该能够完成：

~~~text
定位它是谁
-> 确认 shape/dtype/layout
-> 估算 FLOPs/Bytes/AI
-> 判断理论瓶颈
-> 从 profiler 找实际根因
-> 写出可验证假设
-> 修改并复测
-> 说明算子级和 E2E 收益
~~~

### 本章参考资料

- CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Best Practices：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- PyTorch Custom Operators：https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- Nsight Compute Compute Triage：https://docs.nvidia.com/nsight-compute/ComputeTriage/

---

## 总结：真正要形成的能力

最终目标不是背下某个 API，而是看到一个算子时能连续回答：

1. 它的数学定义是什么？
2. 输入输出的 shape、dtype、layout、stride 是什么？
3. 工作量和并行度如何？
4. 它理论上是 Compute、Memory、Latency 还是 Launch Bound？
5. 数据如何从 HBM 流到计算单元？
6. 哪一级阻止了计算单元持续工作？
7. 修改后如何用正确性、micro benchmark、profiler 和 E2E 证明收益？

这套方法比记住某一个 CUDA Kernel 更容易迁移到 Triton、其他 GPU 和 NPU。

