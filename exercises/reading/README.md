# 陌生代码阅读：先回答，再运行判题

下面是为课程编写的简化片段，用 CUDA/Triton 风格表达，故意保留待审问题；不是截取生产源码，也不是可独立编译的程序。每段五问，围绕读写责任、同步、mask、资源与错误。将 A/B/C 填入 [student.json](student.json)，运行：

~~~powershell
python exercises/reading/check.py
~~~

答错只报告题号，不自动展开答案。完成后可对照 solutions.json，再去读官方完整实现。判题只验证选择；每题还应写一句依据。

## R1：前两个元素的和

前提：一个 block，256 个线程，1<=n<=256。任务只需在 n>=2 时计算 x[0]+x[1]。

~~~cpp
__shared__ float s[256];
int t = threadIdx.x;
if (t < n) {
    s[t] = x[t];
    __syncthreads();
}
if (t == 0 && n >= 2) out[0] = s[0] + s[1];
~~~

1. 谁写 out[0]？A 所有线程；B 线程 0；C 每个 warp 的 lane 0。
2. 线程 0 从哪里取得线程 1 搬来的值？A shared s[1]；B 自动读取线程 1 的寄存器；C host 内存。
3. n=2 时屏障怎么改？A 删除所有同步；B 只在线程 0 同步；C 让 block 所有线程一致到达加载后的屏障。
4. 这个声明占多少 shared 字节？A 256；B 1024；C 8192。
5. 原片段最直接的同步问题是什么？A block 间原子冲突；B 屏障在非一致条件分支内；C 浮点方差不稳定。

## R2：一行 Softmax

前提：BLOCK>=width，BLOCK 为该实现使用的补齐长度；只考虑有限实数输入。

~~~python
row = tl.program_id(0)
col = tl.arange(0, BLOCK)
x = tl.load(X + row * stride + col, col < width, other=0.0)
e = tl.exp(x - tl.max(x, axis=0))
e = tl.where(col < width, e, 0.0)
y = e / tl.sum(e, axis=0)
tl.store(Y + row * width + col, y, col < width)
~~~

1. 一个 program 负责什么？A 整个 batch；B 一个输入标量；C 一行。
2. 能否把它理解为“每个输出与其他列完全独立”？A 能；B 不能，max/sum 有行内依赖；C 只在 FP16 时能。
3. load 的 other 应用什么来保证 max 中性？A 0；B 正无穷；C 负无穷。
4. 增大 BLOCK 可以无代价处理任意长行吗？A 可以；B 不可以，活跃值/资源与无效工作会变化；C 只要输出 mask 就可以。
5. 哪个输入最能击中这段代码的问题？A [0,0]；B [1,2]；C [-1000,-1001] 且 BLOCK>width。

R2 的 sum mask 已存在，问题不是“padding 参与了最终分母”，而是选择 max 基准后失去稳定性保证。回答时区分这两种错误。

## R3：尝试四元素加载

前提：base 是 16 字节对齐的 FP32 矩阵起点，行长 K，row/col 由调用方给出。

~~~cpp
const float* p = base + row * K + col;
float4 value = *reinterpret_cast<const float4*>(p);
if (col < K) consume(value);
~~~

1. 此 load 一次读取几个 float？A 1；B 4；C 16。
2. 加 __syncthreads 能修复地址未对齐吗？A 能；B 不能；C 只要 block 有 32 线程就能。
3. 若 K=10、col=8，只有 col<K 是否足够？A 足够；B 不够，需要整个四元素区间合法；C 写回有 mask 就足够。
4. K=41、row=1、col=0 时，地址相对 base 偏移模 16 为多少？A 0；B 4；C 8。
5. 一个完整候选应该如何处理不满足条件的片段？A 继续强制向量 load；B 忽略最后几个数；C 使用合法的标量/其他回退路径，且先检查再读取。

## 判题与后续阅读

~~~powershell
python exercises/reading/check.py --answers exercises/reading/solutions.json
~~~

参考解应得 15/15。接着阅读 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[Triton Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，把这五种问题应用到真实代码。不要把本页简化片段当成生产实现的全部约束。
