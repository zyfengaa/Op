# Vector Add 精讲：第一次把 CPU 循环搬到 GPU

本篇对应 [vector_add.cu](vector_add.cu) 和 [vector_add_triton.py](vector_add_triton.py)。请先运行无 GPU 的 [elementwise_lab.py](../../examples/cpu/elementwise_lab.py)，再进入 CUDA 代码。若链接相对路径不正确，课程入口仍可从仓库根目录 examples/cpu 查找。

## 1. 从一组具体输入开始

~~~text
A = [1.0, -2.0, 3.0, 0.5, 7.0]
B = [4.0,  5.0,-6.0, 2.0,-1.0]
C = [5.0,  3.0,-3.0, 2.5, 6.0]
~~~

这就是数学语义。每个 C[i] 只依赖 A[i]、B[i]，输出元素之间没有依赖，可以分给不同线程。

接口约定：A、B 长度相等，FP32，输出也是 FP32，A/B 不被修改。若要支持空数组，应由 host 在发射前处理 N=0。

## 2. CPU reference 为什么必须先写

~~~cpp
for (int i = 0; i < n; ++i) {
  c[i] = a[i] + b[i];
}
~~~

这个循环读起来无歧义，可作为比较依据。进入 GPU 后你可能改 block size、数据类型、vector width；reference 不跟着改算法。若 GPU 与 reference 不符，先看第一处不同的索引和值，而不是直接看性能。

## 3. Kernel 的每一行在做什么

~~~cpp
__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
~~~

- __global__ 标记由 host 发起、device 执行的函数。
- const float* 表示输入只读；float* c 是输出 buffer。
- blockIdx.x 标识第几个 block，blockDim.x 是每个 block 的线程数，threadIdx.x 是该 block 内线程编号。
- i 是逻辑元素编号。每个有效线程只写 C[i]，不存在两个线程同时写同一输出的竞争。
- 边界判断必须保护读取和写回。N 不是 block 大小倍数时，最后一个 block 会有无效线程。

### 手算一遍

N=5、block=4 时，grid=ceil(5/4)=2。block0 的四个线程负责 0～3；block1 的线程 0 负责 4，线程 1～3 的 i 分别是 5、6、7，全部跳过。

## 4. Host 代码的完整数据路径

对应 C++ 程序按以下顺序执行：

~~~text
CPU std::vector A/B/C
  → cudaMalloc 得到 d_a/d_b/d_c
  → cudaMemcpy H2D
  → vector_add_kernel<<<grid, block>>>
  → cudaGetLastError + cudaDeviceSynchronize
  → cudaMemcpy D2H
  → 与 CPU 公式逐元素比较
  → cudaFree
~~~

cudaMalloc 只分配地址，不填入 A/B 值；不能跳过 H2D。kernel launch 通常是异步的，host 不能把“已经发起”误当“已经完成”。cudaDeviceSynchronize 在教学程序中确保能在取回前发现执行错误；生产流水线可依赖同一 stream 的先后顺序并采用更合适的同步点。

### 为什么要检查每个 API

若 cudaMalloc 失败却继续使用空指针，后续报错会让问题来源更难定位。源码中的 CUDA_CHECK 在每个调用处输出具体失败表达式，缩短排障链路。

## 5. 运行与预期

在仓库根目录运行：

~~~powershell
cmake -S examples/cuda -B build/cuda
cmake --build build/cuda --config Release
.\build\cuda\Release\vector_add.exe
~~~

Windows 多配置生成器可能在 Release 下；Linux 通常执行 build/cuda/vector_add。示例固定 N=100003，预期末行包含：

~~~text
vector add correctness: PASS
~~~

如果 GPU 不可用，先执行：

~~~powershell
python examples/cpu/elementwise_lab.py
~~~

预期 add 行为 [5.0, 3.0, -3.0]。CPU 例子只是语义练习，不是 GPU 性能测试。

## 6. 错误案例：从症状反推原因

### 错误 A：grid 用 floor division

~~~cpp
int grid = n / block;
~~~

N=5、block=4 时只发一个 block，C[4] 没有线程写，常表现为最后几个结果保持原值或随机值。修正为 (n+block-1)/block。

### 错误 B：删除 if

最后一个 block 的无效线程会访问 A/B/C 越界。即使一次运行“看起来正常”，也不能认为安全；使用 Compute Sanitizer 或边界测试验证。

### 错误 C：只测一次 wall clock

CPU 计时器只围住 launch 时，测到的可能主要是排队时间。测 kernel 可使用 CUDA Event 并重复运行；测应用 E2E 要把真实的传输和后处理纳入同一口径。两组数据不可混成一个速度结论。

### 错误 D：把空向量发成 0 grid

N=0 时应在 host 端直接返回空输出。教学代码使用固定正 N，所以如果要做通用 API，这一步必须补齐。

## 7. 性能计算先手算

FP32 版本每个元素读 A 的 4 字节、读 B 的 4 字节、写 C 的 4 字节。最小逻辑流量约 12N 字节，运算量 N 次加法，所以 AI≈1/12 FLOP/Byte。

以 N=1,000,000 为例，逻辑流量约 12 MB。若 kernel 实测为 20 微秒，有效带宽约 600 GB/s。这只是算式示例，不代表仓库实测。若测得 60 微秒，约 200 GB/s。比较之前要固定输入位置和 GPU 运行状态。

### 为什么加 shared memory 通常没用

每个 A[i]、B[i] 只被消费一次，没有明显重用。搬到 shared memory 再读出多一道操作，可能反而更慢。这个判断由数据依赖推导；不是“所有 kernel 都要用 shared memory”的固定套路。

## 8. 两个进阶小案例

1. ReLU：Y[i]=max(X[i],0)。保留同一索引/边界结构，替换算术表达式。输入 [-2,0,3] 输出 [0,0,3]。
2. Bias+ReLU 融合：Y[i]=max(X[i]+bias[i],0)。比较分两次 launch 与一次 launch 的中间写回/读取，注意 bias 可能是一维广播，不总与 X 同 shape。

## 9. 练习及答案要点

1. N=257、block=256：grid=2；第二个 block 只有一个有效线程。
2. FP16 版本：逻辑最小流量变为约 6N 字节，但计算和累加精度、向量化、设备支持需要重新定义。
3. 将两个输入改为不同长度时，host 应拒绝并给出清楚错误，不应让 kernel 猜测哪一边可读。

## 10. 本章通过标准

你能手算任意 N 的 grid/tail，解释 host→device→host 的完整路径，指出 kernel-only 与 E2E 的测量边界，并给出至少两个会造成错误的反例。

## 参考资料

- 必读：[CUDA Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。
- 必读：[CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。
- 查阅：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
- 对照：[Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/)。
