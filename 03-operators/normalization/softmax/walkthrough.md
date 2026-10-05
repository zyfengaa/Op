# Softmax 完整推导：从一行数字到 Online 归约

本章的目标是能手算稳定 Softmax、推导分块合并公式、解释 padding mask，并在没有 GPU 时跑通 [CPU 实验](../../../assets/cpu/softmax_lab.py)。

## 1. 语义和真实数字

对 x=[1000,1001,999]，直接计算 exp(1000) 在普通浮点环境中会溢出。减去行最大值 1001 后，得到 [-1,0,-2]。指数约为 [0.367879,1,0.135335]，和约 1.503214，输出约为 [0.244728,0.665241,0.090031]，和为 1。平移不改变比值，因为分子分母都乘以相同的 exp(-m)。

这说明 max 不是可省略的装饰步骤，而是正确性的一部分。每个输出都依赖全行 max 和全行 sum，因此不像 ReLU 那样每元素完全独立。

## 2. 从朴素版本到一行一个 block

最清楚的 CPU reference 分三遍：先找最大值 m；再计算 e[i]=exp(x[i]-m) 与 l=sum(e)；最后 y[i]=e[i]/l。GPU 的最简单映射也可用三个 kernel，但会写读中间数据并产生多个 launch。把一行交给一个 block 后，线程共同做 max reduction，所有线程看到 m 后算 e，再做 sum reduction，最后写 y。两个 reduction 阶段之间需要正确同步；不能让部分线程提前覆盖共享存储。

若行宽 5、物理 block 宽 8，max 阶段的三个无效 lane 应使用 -∞，sum 阶段应贡献 0。max 阶段用 0 会改变全负输入的最大值并失去稳定性保证；并非每个全负输入的概率都会因此改变，具体反例见第 9 节。sum 阶段若把无效元素计入，归一化的数据集合就会改变。

空行或整行被 mask 的行为必须在接口里单独规定；不能默认分母不为 0。混合精度中可选择 FP32 累加，最终再转换到目标输出 dtype，容差应按此路径确定。

## 3. Online Softmax 为什么能分块

假设旧块统计为 m_old=max(旧值)，l_old=Σexp(x-m_old)。新块统计为 m_blk、l_blk。合并时：

~~~text
m_new = max(m_old, m_blk)
l_new = exp(m_old-m_new)*l_old + exp(m_blk-m_new)*l_blk
~~~

两个系数只是把各自的“参考最大值”统一到 m_new。举例，旧块 [1000,1001] 的 m_old=1001，l_old=e^-1+1≈1.367879；新块 [999] 的 m_blk=999，l_blk=1；合并后 m_new=1001，l_new=1.367879+e^-2≈1.503214。与整行直接算完全一致。

如果只要输出概率，最终仍需得到各个 x 对应的 exp(x-m)/l；可以重新读取输入，或在可用资源内保留中间值。Online 统计并不自动等于“只读一次输入”，更不自动等于 FlashAttention。后者还维护按概率加权的 V 累加器。

## 4. 运行、验证和扩展

~~~powershell
python assets/cpu/softmax_lab.py
python -m unittest discover -s assets/cpu
~~~

将 block_size 改成 1、2、大于行长，比较稳定 reference 与 online 结果；测试全负、极大值、重复最大值、非 2 的幂行长。断言每个概率在 [0,1] 内、概率和接近 1、整体平移常数后输出不变。相对误差在接近 0 的位置不可靠，应同时看绝对误差。GPU 测试还要测行宽 31/32/33 和 1023/1024/1025，抓住末块边界。

性能实验记录行数、行宽、dtype、布局、kernel launch 数、总耗时和读写字节。极短行可能由 launch 或 block 利用率主导；极长行可能需要跨 block 分段。是否融合必须由数据决定。

## 5. 练习与答案要点

练习 A：对 [-5,-4,-3] 求稳定 softmax。减去 -3，得 [e^-2,e^-1,1] 再除以 1.503214，输出约 [0.090031,0.244728,0.665241]。练习 B：为什么 padding 的 max 单位元是 -∞？因为 max(x,-∞)=x；sum 的单位元是 0。练习 C：把 [1000,1001,999] 分成 [1000] 和 [1001,999] 两块，用上式合并，所得 m=1001、l≈1.503214。

## 6. 把一行数据分给四个线程，逐项跟踪中间量

前面的公式没有说明线程究竟负责什么。现在用一个可以运行的缩小模型把这一步展开。输入改为五个数：

~~~text
x = [1000, 1001, 999, 998, 1002]
D = 5
T = 4
~~~

一个 block 负责这一整行；线程 t 访问 t、t+T、t+2T，直到列号不再小于 D。这种分配允许一行比线程数长，也允许有线程没有元素。

| 线程 t | 负责列号 | 读取的值 | 线程局部最大值 |
|---:|---|---|---:|
| 0 | 0,4 | 1000,1002 | 1002 |
| 1 | 1 | 1001 | 1001 |
| 2 | 2 | 999 | 999 |
| 3 | 3 | 998 | 998 |

四个局部最大值放入共享数组，树形归约得到 1002。接着线程仍然处理自己的列，但计算内容变为 exp(x[col]-1002)：

| 线程 t | 指数贡献 | 局部和（约） |
|---:|---|---:|
| 0 | exp(-2)+exp(0) | 1.135335 |
| 1 | exp(-1) | 0.367879 |
| 2 | exp(-3) | 0.049787 |
| 3 | exp(-4) | 0.018316 |

再归约得到分母约 1.571317。最后每个线程回到原来的列写输出。注意线程 0 写的是输出列 0 和 4，绝不是把它的“局部和”当成某个概率写出去。

运行下面的命令会打印上述列归属、局部最大值、局部和及最终输出：

~~~powershell
python assets/cpu/dataflow_lab.py --topic softmax
~~~

程序使用四个逻辑线程是为了便于手算；CUDA 示例使用 256 个真实线程。改变 T 会改变工作分配和归约顺序，但不应改变算子的数学语义。

## 7. 对照 CUDA 实现逐段阅读

完整宿主程序、CPU reference 和 kernel 位于 [row_ops.cu](../../../assets/cuda/row_ops.cu)。先找到 row_softmax：

~~~cpp
const int t = threadIdx.x;
const size_t base = static_cast<size_t>(blockIdx.x) * width;

float local_max = -CUDART_INF_F;
for (int col = t; col < width; col += blockDim.x) {
  local_max = fmaxf(local_max, x[base + col]);
}
const float m = block_reduce<true>(local_max, scratch);
~~~

blockIdx.x 选择第几行，base 是这一行的起始元素偏移。width 是真实行长，不是向上补齐的长度。局部最大值初始化为负无穷，使没有分到元素的线程也能参加后面的归约。所有线程必须调用 block_reduce；不能对 t>=width 的线程提前 return。

接着看第二阶段：

~~~cpp
float local_sum = 0.0f;
for (int col = t; col < width; col += blockDim.x) {
  local_sum += expf(x[base + col] - m);
}
const float denominator = block_reduce<false>(local_sum, scratch);
~~~

这段重新读取 x。这样写用更多读取换来了直观的代码和较小的每线程存储。它是“一次 kernel launch 中完成全部阶段”，并不是“输入只读取一次”。若将每个线程的多个 x 保存在寄存器中，可以减少读取，但行长越大，寄存器占用和编译展开的问题越明显。

最后一段再次计算 expf 并写输出。这同样有重复计算，却把“先保证结果”和“进一步减少读算”分成了可观察的两个版本。理解 baseline 后，才适合实现缓存指数值的优化版。

## 8. 为什么 reduction helper 的末尾还需要一次同步

共享数组 scratch 会先存局部最大值，下一次调用又要存局部和。假设归约完成后只写：

~~~cpp
return scratch[0];
~~~

快线程可能已经返回、算完下一阶段，并把新的 local_sum 写入 scratch[0]；慢线程此时才读取上一轮的 scratch[0]，就把“局部和”误当成了“全行最大值”。这类错误只在特定调度下出现，普通小测试可能碰巧通过。

本仓库的 helper 按以下顺序结束：

~~~cpp
const float result = s[0];
__syncthreads();
return result;
~~~

每个线程先把归约结果读入自己的局部变量；屏障确保所有线程都读完，才允许下一次调用覆盖共享存储。理解这个屏障的理由，比记住“归约循环里面要同步”更重要。

## 9. padding 为 0 到底在什么情况下出错

max 的中性值确实应为 -∞，但“全负输入 padding=0 就一定算出错误概率”这个说法不精确。对 [-5,-4,-3]，若把平移基准错误地变成 0，同时 sum 仍然只累加有效元素，精确数学中的 softmax 依旧相同，因为任意共同平移都保持比值。

问题在数值保证：对 [-1000,-1001]，使用 0 做基准会计算 exp(-1000)、exp(-1001)，两者在普通双精度下都可能下溢为 0；分母也变成 0。真正最大值 -1000 则得到 [1,exp(-1)]，结果稳定。

另一种错误是把 padding 本身也送入指数和。例如真实输入 [-1,-2]，补一个 0 后直接对三个元素做 softmax，再丢掉第三个输出；保留下来的两个概率和会小于 1。这里是输入集合变了，而不只是平移基准变了。

因此应分别审查 max mask、sum mask 和输出 store mask，不能只看是否“用了 mask”。

## 10. Online 合并不是魔法：从求和式一步步变形

旧数据集合 A 的状态为 mA 和 lA，其中：

~~~text
lA = Σ(i∈A) exp(xi-mA)
~~~

现在要用新基准 m 来表示同一批旧数据：

~~~text
Σ exp(xi-m)
= Σ exp((xi-mA)+(mA-m))
= exp(mA-m) × Σ exp(xi-mA)
= exp(mA-m) × lA
~~~

新块 B 完全同理，所以两块相加得到前文公式。取 m=max(mA,mB) 是为了保证所有指数的输入非正；算法正确性来自换基准，数值稳定性来自选择最大值。

故障练习：先处理 [0]，再处理 [10]。若仅做 l=l_old+exp(10-10)，得到 l=2，从而把第二项的概率算成 0.5。正确旧贡献应是 exp(0-10)，所以 l≈1.000045，第二项概率≈0.999955。这个例子专门用于发现“最大值改变时忘记缩放旧统计”的错误。

## 11. 编译、执行和认识输出

在已有兼容 CUDA Toolkit、宿主 C++ 编译器和 NVIDIA GPU 的环境中：

~~~powershell
cmake -S assets/cuda -B build/cuda
cmake --build build/cuda --config Release
.\build\cuda\Release\row_ops.exe
.\build\cuda\Release\row_ops.exe --benchmark
~~~

单配置生成器通常使用 build/cuda/row_ops；Linux 执行 ./build/cuda/row_ops。程序默认同时测试 Softmax 和 LayerNorm。每个宽度有三种行：普通变化值、常量值、1000 附近的值；宽度包含 1、31/32/33、255/256/257、1023/1024/1025、4097。

通过时打印算子名、rows、width、PASS 和最大绝对误差；错误时打印首个失败位置、实际值和参考值并返回非零。示例输出格式如下，尖括号里的数值必须以实际运行结果为准：

~~~text
softmax rows=3 width=257 PASS max_abs_error=<measured>
~~~

--benchmark 额外预热 20 次，以 CUDA events 测量 100 次 launch 的批平均设备时间。该时间不包含分配与主机/设备复制，也不是整条模型推理时间。本机若没有 CUDA 环境，仍可运行上面的 CPU 数据流实验，但不能据此填写 GPU 性能表。

## 12. 优化实验应该怎样有因果关系

先固定基线的形状与误差，然后只改变一个主要因素：

| 改动 | 希望减少的成本 | 新增的代价 | 验证方式 |
|---|---|---|---|
| 缓存每线程读入的 x | 输入重读 | 寄存器/局部存储 | 看生成代码与实际时间 |
| 缓存 exp(x-m) | 重算指数 | 更多活跃值 | 检查寄存器和 spill |
| 一个 block 处理多行 | 短行线程浪费 | 更复杂的行内归约 | 重点测短行、多行 |
| 一行使用多个 block | 很长行的并行度不足 | 多阶段合并和 launch | 同时测完整路径 |
| warp 级归约 | 共享存储和屏障 | lane mask/平台假设 | 先补边界与同步测试 |

输出的理想最小流量是读 x、写 y，即 FP32 下约 8BD 字节。本 baseline 在源码层面读 x 三遍、写 y 一遍，约 16BD 字节；实际 HBM 流量还受 cache 影响。用 8BD/time 算出来的是“有效带宽”，不能冒充 profiler 的 HBM 测量结果。

## 13. 本章交付物与自检

提交一份列归属表、一份包括非整除宽度的 correctness 日志、一份明确计时范围的性能记录，并回答：

1. 为什么 D=5、T=256 时有 251 个线程仍要参加屏障？
2. 为什么 block_reduce 返回前先复制 s[0] 再同步？
3. 输入读三遍的 fused kernel 与三次 kernel launch 有何区别？
4. 若 m 从 0 变为 10，旧分母乘什么系数？
5. padding 的 max 中性值错误和把 padding 加入分母，是否是同一种错误？

答案依次是：屏障和归约覆盖整个 block；保护共享数据的最后一次读取；launch 次数和内存访问次数是两个维度；乘 exp(-10)；不是，前者破坏稳定性保证，后者改变参与归一化的数据集合。

## 参考资料（按学习顺序）

1. [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：观察一行到 block 的映射和 mask。
2. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：同步与内存模型。
3. [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：Online 统计在 Attention 中的延伸。
