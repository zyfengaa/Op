# 基础精讲：从数组地址到 GPU 线程

这篇是第 00 阶段的完整练习。读完应能在纸上算出线程负责哪个元素、该元素在内存中的偏移，并能判断哪些成本不属于算子计算本身。

## 1. 一个数组到底是什么

Python 的二维列表容易让人以为矩阵在内存里天然是“二维的”。GPU kernel 实际拿到的是一段地址。对连续 row-major 矩阵，元素 A[row,col] 的线性位置为 row×cols+col。

以三行四列为例：

~~~text
row 0:  0  1  2  3
row 1:  4  5  6  7
row 2:  8  9 10 11
~~~

A[2,1] 是第 2 行第 1 列，其偏移为 2×4+1=9。如果元素是 FP32，每个占 4 字节，字节偏移为 9×4=36。实际地址等于基地址加 36 字节。

### 1.1 为什么还要有 stride

若从一个宽度为 8 的大矩阵中取出每行前 4 列，视图的逻辑 shape 可以是 [3,4]，但下一行从 8 个元素后开始。此时行 stride 为 8，A[2,1] 的偏移是 2×8+1=17，而不是 9。算子如果仅凭 shape 猜 stride，就会从错误位置取值。

运行 Python 实验：

~~~powershell
python assets/cpu/indexing_lab.py
~~~

其中 A[2,3] 在连续宽度 5 的矩阵里偏移 13；在行 stride 为 8 的视图里偏移 19。请先独立算，再看输出。

## 2. 线程映射如何从公式得来

Vector Add 的输出彼此独立：C[i]=A[i]+B[i]。最自然的分工是一个线程负责一个 i。CUDA 的 block 和 grid 把全局编号拆成两级：

~~~text
i = blockIdx.x × blockDim.x + threadIdx.x
~~~

对 N=10、blockDim.x=4，grid=ceil(10/4)=3：

| blockIdx.x | threadIdx.x | i | 是否读写 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 是 |
| 0 | 3 | 3 | 是 |
| 1 | 0 | 4 | 是 |
| 1 | 3 | 7 | 是 |
| 2 | 0 | 8 | 是 |
| 2 | 1 | 9 | 是 |
| 2 | 2 | 10 | 否 |
| 2 | 3 | 11 | 否 |

GPU 启动 12 个逻辑线程，但只有 10 个有效元素。最后两个线程必须在读取 A/B 和写 C 前检查 i<N。只在写回时检查还不够，因为越界读取本身就不安全。

### 2.1 算 grid 的正确方式

整除的 floor 版本 grid=N/block 会漏掉尾部。例如 N=10、block=4 算出 2 个 block，只覆盖 0～7。ceil division：(N+block-1)/block=3，覆盖 0～11，再通过 mask 去掉无效的 10、11。

空输入 N=0 时 grid=0；有些运行时不允许发射 0 block，host 端应直接返回空输出或跳过 launch。这是接口语义，不是 kernel 内 if 能解决的问题。

## 3. 二维映射与地址

矩阵加法 C[r,c]=A[r,c]+B[r,c] 可以用二维 block：

~~~text
r = blockIdx.y × blockDim.y + threadIdx.y
c = blockIdx.x × blockDim.x + threadIdx.x
offset = r×stride + c
~~~

判断顺序：先检查 r<M、c<N，再计算可访问地址。x 映射列通常有利于相邻线程访问连续地址，但这仍取决于 layout、编译结果和设备。

### 3.1 一个常见错误

错误写法：offset=r×M+c。对于 M 行 N 列的连续 row-major 矩阵，行跨度是 N，不是 M。例如 M=2、N=3，A[1,0] 应在偏移 3，误用 M 得到 2。

## 4. 为什么 GPU 未必更快

用一个简单成本模型：

~~~text
CPU 方案时间 ≈ CPU 计算时间
GPU 方案时间 ≈ H2D + launch + GPU kernel + D2H + 可能的同步
~~~

如果输入已在 GPU，H2D/D2H 可以不发生；如果每次都在 CPU 上产生数据并立即取回，小数组的搬运和启动可能压倒计算收益。

这个模型不能给出固定阈值。你需要分别量测 kernel-only 和 end-to-end，并说明数据最初在哪里、最终消费者在哪里。

## 5. 内存层级的三个问题

看到一个慢 kernel，先问：

1. 数据总共要读写多少字节？
2. 有多少次重复使用？能否停留在 shared memory 或 register？
3. 数据是连续访问、跨步访问，还是有 bank conflict？

以 Vector Add 为例，每个 A[i] 和 B[i] 只用一次；shared memory 通常不能创造新的复用。以 GEMM 为例，一个 A[m,k] 可参与许多 n 列结果；一个 B[k,n] 可参与许多 m 行结果，tiling 才有复用空间。

## 6. 三个小练习及答案

### 练习 A

N=37、block=16，求 grid、总线程、无效线程。答案：ceil(37/16)=3 个 block，共 48 线程，无效 11 个。有效下标为 0～36。

### 练习 B

row-major 矩阵 [5,7]，A[3,2] 的元素偏移是多少？答案：3×7+2=23。若 FP32，字节偏移 92。

### 练习 C

为什么 block 内屏障不能同步整个 grid？答案：普通 CUDA 线程块可独立调度，一个 block 内的线程可以在当前 SM 共享状态并同步；不同 block 可能尚未被调度。跨 block 的全局依赖通常拆成多个 kernel 或使用专门机制。

## 7. 完成标准

能够不用运行代码算出上面所有下标；能解释连续矩阵和视图的 stride；能说明 GPU 时间由哪些阶段构成；能指出哪些结论需由 profiler 验证。完成后再进入 Vector Add。

## 参考资料（按优先级）

1. 必读：[CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)——对应本章线程、block、grid。
2. 必读：[Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)——对应全局索引和边界。
3. 查阅：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)——对应内存与性能的初步判断。
4. 进阶：[CMU 15-418](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s20/www/)——系统理解并行架构。
