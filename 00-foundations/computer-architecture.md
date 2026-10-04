# CPU、GPU 与并行执行模型

## 1. 为什么 GPU 适合大量相似工作

CPU 通常用少量复杂核心处理低延迟任务和复杂分支；GPU 用大量执行单元并行处理相似任务。GPU 的优势不意味着任何程序放到 GPU 都会更快：数据搬运、kernel 启动、可用并行度和同步成本都可能抵消计算收益。

可以把加法作为对照：CPU 循环可能只需几十纳秒级的局部工作，而 GPU 调用包含 launch 和可能的数据拷贝。只有工作量足够大、数据已在 GPU，或者多个操作可以融合时，GPU 才容易体现优势。

## 2. SIMD 与 SIMT 的直观区别

- SIMD：一条指令对多个数据元素执行同一种操作。
- SIMT：程序员描述许多逻辑线程，硬件把线程组织成组执行。

CUDA 常用术语是 thread、warp、block、grid。其他平台名称和约束可能不同；学习时应先掌握“很多独立索引处理很多元素”这个共性，再学具体 API。

## 3. 用索引推导线程工作

对于长度 N 的向量：

```text
global_id = block_id * threads_per_block + thread_id
```

假设每个 block 4 个线程、N=10：

| block | thread | global_id | 是否有效 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 是 |
| 0 | 3 | 3 | 是 |
| 1 | 0 | 4 | 是 |
| 2 | 1 | 9 | 是 |
| 2 | 2 | 10 | 否，必须屏蔽 |

对于二维输出矩阵，通常用二维 grid 映射 row/col，再用 `row * stride + col` 得到地址。映射方案是算法设计，不是硬件自动猜出来的。

## 4. Block、SM 与同步范围

CUDA block 是调度和协作的常见单位。一个 block 在一个 SM 上执行；同一 block 的线程可通过 shared memory 协作，并用 `__syncthreads()` 建立 block 内屏障。普通 kernel 没有通用的 grid 级屏障，因此跨 block reduction 通常需要第二次 kernel、原子操作或专门的 cooperative launch 机制。

不要把 warp 宽度、SM 数量或每 SM 资源写死在跨平台代码里。运行时查询和平台文档才是依据。

## 5. 分支和发散

若同一 warp 的线程走不同分支，硬件可能需要分别执行多个路径。尾块中的 `if (idx < N)` 通常是必要边界保护，不能为了消除分支而冒险越界。要区分“分支存在”和“分支是瓶颈”；只有 profiling 表明控制流显著影响性能时才优化。

## 6. 小练习

1. 对 N=37、block=16，计算 grid 大小和无效线程数。
2. 对矩阵 M=5、N=7，列出 row-major 中 `A[3,2]` 的线性偏移。
3. 解释为什么不同 block 不能用普通 `__syncthreads()` 同步。
4. 解释为什么将 CUDA warp=32 的假设直接移植到其他 GPU 风险很大。

## 参考资料

- [CUDA Programming Guide · Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Programming Guide · Hardware Implementation](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CMU 15-418 course materials](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s20/www/)
