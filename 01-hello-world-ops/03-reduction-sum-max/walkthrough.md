# Reduction 精讲：为什么求和比逐元素加法难

对应 [CPU 树形归约实验](../../examples/cpu/reduction_lab.py) 和 [CUDA 单 block 示例](block_reduce_sum.cu)。本章从整数手算开始，再解释线程协作和同步。

## 1. 先定义三种算子

~~~text
Sum([x0,...,xN-1]) = Σ xi
Max([x0,...,xN-1]) = 最大 xi
Mean([x0,...,xN-1]) = Sum(x) / N
~~~

对于 N=0，Sum 常定义为 0，但 Max 没有自然最大值，需要接口显式规定是报错、返回 -∞，还是采用框架特殊约定。Mean([]) 也需要明确语义。不要让 GPU 未初始化内存替你做决定。

## 2. 串行 reference

~~~cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i) sum += x[i];
~~~

如果只用一个 GPU 线程照搬这段循环，结果可用于对照，但不会充分利用 GPU 并行度。并行改造必须回答：多个线程各自算出的部分和，谁来合并？

## 3. 五个数手算成八个槽

输入 [1,2,3,4,5]，为讲解二叉树，补零到长度 8：

~~~text
初始  [1,2,3,4,5,0,0,0]
步 1  前 4 个槽分别加后 4 个槽
      [6,2,3,4,5,0,0,0]
步 2  前 2 个槽分别加后 2 个槽
      [9,6,3,4,5,0,0,0]
步 3  第 0 槽加第 1 槽
      [15,6,3,4,5,0,0,0]
~~~

只有第 0 槽的 15 是最终输出。运行：

~~~powershell
python examples/cpu/reduction_lab.py
~~~

看每一步数组状态，再自己用 [3,1,4,1,5] 重做一遍。

## 4. 为什么要用 shared memory

一个 CUDA block 中每个线程先做局部加法，写入 shared memory 的 partial[tid]。之后，线程 0 要读取其他线程写入的值，必须先确认写入都完成。block 屏障 __syncthreads() 就是这个阶段边界。

~~~cpp
partial[tid] = local;
__syncthreads();
for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
  if (tid < offset) partial[tid] += partial[tid + offset];
  __syncthreads();
}
~~~

该教学实现固定 THREADS=256，它是 2 的幂。若换成 300 线程，上面的 halving 逻辑可能漏掉一些元素，不能只改常量。

### 为什么 barrier 放在 if 外面

如果一部分线程执行 __syncthreads()，另一部分线程跳过，整个 block 可能不能继续。教学代码让所有线程都到达每轮 barrier，只有 tid<offset 的线程做加法。

## 5. 每线程先处理多个输入

示例 N=1003、block=256。线程 t 处理 t、t+256、t+512、t+768 等位置，直到超出 N。这样 256 个线程覆盖全部 1003 项，不需要为每个输入发一个线程。之后每线程仅把自己的 local 写入 shared memory。

### 最后几个线程怎样处理尾部

例如线程 250 处理 250、506、762，然后 1018 越界，因此循环停止。它仍会参与 block 内归约，自己的 partial 已被初始化为局部和。尾部没有读越界。

## 6. Sum、Max、Mean 的 padding identity 不一样

- Sum：无效槽应补 0，因为 a+0=a。
- Max：无效槽应补 -∞，因为 max(a,-∞)=a。
- Min：无效槽应补 +∞。
- Mean：局部 sum 可以补 0，但最后要除真实 N，不能除 padding 后的宽度。

错误例：Max([-5,-2,-9]) 若补 0，会输出 0，它根本不在输入里。CPU 脚本专门演示这个问题。

## 7. 多 block 为什么需要第二阶段

当 N 巨大时，单 block 会让一个 SM 处理所有输入，设备其他 SM 没有工作。一般拆成：

~~~text
kernel 1: 多个 block 各自写 partial_sum[blockIdx]
kernel 2: 对 partial_sum 再归约成一个值
~~~

两个 kernel 在同一个 stream 中顺序执行时，第二个能看到第一个完成的结果。普通 __syncthreads() 只能管一个 block，不能让某个 block 等整个 grid。

AtomicAdd 也可合并部分结果，但不同 block 的顺序可能不固定，性能受争用影响。不能以“少了第二个 kernel”就断定它更快。

## 8. 浮点误差从哪里来

浮点数有限精度，(a+b)+c 与 a+(b+c) 可能不同。串行 reference 和树形 reduction 顺序不同，结果可能有小误差。测试时根据 dtype 和数据范围设置 atol/rtol，并记录最大误差。对大量大/小数混合输入，误差可能更明显。

## 9. 何时观察什么指标

- N 很小：launch 与同步可能主导。
- 一个 block 处理巨大 N：并行度不足。
- 多 block atomic 竞争：atomic throughput 或争用。
- 大量 shared access：bank conflict 和同步开销。
- 纯流式 sum：读取带宽、实际 transaction。

先用时间线定位总开销，再查看单 kernel 的访存和 stall。不要只见 occupancy 低就调大 block。

## 10. 错误案例与修正

| 症状 | 可能错误 | 如何确认与修正 |
|---|---|---|
| Max 全负输入输出 0 | 无效 lane 补 0 | 改为 -∞，测试全负 |
| 大输入结果缺一段 | 每线程只处理一次 index | 改为 stride 循环 |
| 结果偶尔变化 | 缺少同步或 atomic 顺序变化 | 加必要 barrier；若需确定性采用固定树 |
| 很大 N 很慢 | 只有一个 block | 多 block partial + 第二阶段 |
| Mean 偏小 | 分母用了 padded width | 除以有效元素数 N |

## 11. 练习与答案要点

1. N=5 补到 8 槽，为什么 sum 补 0？因为 0 是加法 identity。
2. 输入全为负，Max identity 是什么？-∞。
3. 多 block partial 有 1000 个结果，下一个 kernel 的输入是什么？长度 1000 的 partial 数组，而不是原 N 个输入。
4. 为什么 CPU 与 GPU 数值可能有细微差异？加法顺序不同。

## 12. 完成标准

你能写出单 block reduction 的数据流，解释每次 barrier 的目的，能说明单 block 与多 block 的适用范围，并能为 Sum/Max/Mean 选择正确 identity。

## 参考资料

- 必读：[CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)——同步范围。
- 必读：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)——访存与资源。
- 查阅：[CUB DeviceReduce](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)——生产库如何提供 reduction API。
