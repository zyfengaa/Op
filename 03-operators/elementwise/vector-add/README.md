# Vector Add：你的第一个 GPU Kernel

第一次学习请先读 [完整逐步教程](walkthrough.md)，再把本文件当作复习提纲。

理解首次实现后，进入 [完整开发经历：五件套](case-study.md)。那里有实际三版本 CUDA 源码、四个可自动判题的故障、原始 CPU profiler 报告和迁移实践，不以看懂这个提纲作为结束。

## 目标与问题定义

给两个长度为 N 的 FP32 向量 A、B，计算 C：

~~~text
C[i] = A[i] + B[i], 0 <= i < N
~~~

输入输出 shape 都是 `[N]`，不原地修改输入。算法每个元素需要读 A、读 B、写 C，共约 12 bytes 和 1 次加法（FP32）。

## CPU reference

~~~cpp
for (int i = 0; i < n; ++i) {
  c[i] = a[i] + b[i];
}
~~~

CPU 版本定义语义。GPU 结果逐元素与它比较，浮点数使用 tolerance。

## CUDA kernel 逐步读

源码：[vector_add.cu](impl/vector_add.cu)
Triton 对照：[vector_add_triton.py](impl/vector_add_triton.py)

核心代码：

~~~cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
~~~

- `blockIdx.x`：当前 block 在 grid 中的编号。
- `blockDim.x`：每个 block 的线程数。
- `threadIdx.x`：线程在 block 内的位置。
- 三者算出全局元素编号 `i`。
- 最后的 if 是 tail mask，保护长度不是 block 整数倍的输入。

## Host 到 Device

最基础的完整流程：

1. CPU 创建并填充 `std::vector<float>`。
2. `cudaMalloc` 申请 GPU buffer。
3. `cudaMemcpy(...HostToDevice)` 传入数据。
4. launch kernel。
5. `cudaGetLastError` 检查 launch，`cudaDeviceSynchronize` 等待执行。
6. `cudaMemcpy(...DeviceToHost)` 取回结果。
7. 与 CPU reference 对比并释放显存。

示例程序按这个流程实现，并使用 `N=100003` 验证尾部线程。

## 编译和运行

在有 CUDA Toolkit 和兼容 NVIDIA GPU 的环境中：

~~~powershell
cmake -S assets/cuda -B build/cuda
cmake --build build/cuda --config Release
~~~

Windows 执行 `build/cuda/Release/vector_add.exe`；Linux 执行 `build/cuda/vector_add`。运行结果应包含 `vector add correctness: PASS`。

## 常见错误

1. `grid = n / block`：当 N 有余数时漏算尾部，应使用 `(n + block - 1) / block`。
2. 删除 `i < n`：最后一个 block 越界读写。
3. 只调用 launch 不检查错误：kernel 是异步的，错误可能在后续同步点才出现。
4. 用 CPU wall clock 直接包围 launch：测到的可能只是排队时间，不是 kernel 完成时间。
5. 小输入拷贝后测 kernel：launch/同步成本可能远大于加法本身。

## 性能怎么读

理想传输量约 `3 * N * sizeof(float)`。有效带宽：

~~~text
effective_bandwidth = bytes / seconds
~~~

Vector Add 通常是 Memory Bound。要先区分只测 kernel 的时间和包含 H2D/D2H 的端到端时间。设备数据已常驻 GPU 的推理 workload 不应把每次 host copy 假设成必然成本。

## 练习

1. 将 N 改为 1、31、32、33、255、256、257 和 100003，解释尾块行为。
2. 将输入改成 `C[i] = alpha*A[i] + B[i]`，重新写 reference。
3. 设计一个计时器，分别报告 kernel 时间和完整 copy + kernel 时间。
4. 让 AI 生成另一种 block size 配置，自己解释并在目标 GPU 上测量它，不直接接受“更快”的断言。

## 参考资料

- [CUDA Programming Guide · Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [CUDA Best Practices · Device Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-accesses)
- [Triton Vector Add Tutorial](https://triton-lang.org/main/getting-started/tutorials/)
