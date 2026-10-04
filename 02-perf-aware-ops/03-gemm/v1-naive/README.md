# GEMM V1：Naive CUDA baseline

源码：[naive_gemm.cu](naive_gemm.cu)。CPU reference、显存分配、kernel launch 和结果比较都在最小程序中。

## 线程映射

二维 block 中每个线程负责一个输出 C[row,col]，沿 K 串行累加。grid 覆盖 M/N 输出平面，边界条件屏蔽非整除尺寸。

## 为什么慢

不同输出线程会重复读取相同 A/B 元素。同一行的相邻输出会重复读取 A；同一列的多个输出会重复读取 B。Naive 版本没有显式 tile 复用，不是高性能 GEMM。

## 编译、运行和验证

从仓库根目录配置 examples/cuda 构建。测试尺寸为 37×41 乘 41×29，刻意让尾部不整齐；输出应在容差范围内匹配 CPU reference。

## 需要做的实验

- 用 CUDA Event 计 kernel 时间，不用单次 host wall clock。
- 增大 M/N/K 到目标设备上的代表性尺寸。
- 记录 GFLOPS，并说明它为何低于峰值。
- 用 Compute Sanitizer 检查越界和未初始化访问。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
