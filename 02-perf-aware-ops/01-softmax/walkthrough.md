# Softmax 完整推导：从一行数字到 Online 归约

本章的目标是能手算稳定 Softmax、推导分块合并公式、解释 padding mask，并在没有 GPU 时跑通 [CPU 实验](../../examples/cpu/softmax_lab.py)。

## 1. 语义和真实数字

对 x=[1000,1001,999]，直接计算 exp(1000) 在普通浮点环境中会溢出。减去行最大值 1001 后，得到 [-1,0,-2]。指数约为 [0.367879,1,0.135335]，和约 1.503214，输出约为 [0.244728,0.665241,0.090031]，和为 1。平移不改变比值，因为分子分母都乘以相同的 exp(-m)。

这说明 max 不是可省略的装饰步骤，而是正确性的一部分。每个输出都依赖全行 max 和全行 sum，因此不像 ReLU 那样每元素完全独立。

## 2. 从朴素版本到一行一个 block

最清楚的 CPU reference 分三遍：先找最大值 m；再计算 e[i]=exp(x[i]-m) 与 l=sum(e)；最后 y[i]=e[i]/l。GPU 的最简单映射也可用三个 kernel，但会写读中间数据并产生多个 launch。把一行交给一个 block 后，线程共同做 max reduction，所有线程看到 m 后算 e，再做 sum reduction，最后写 y。两个 reduction 阶段之间需要正确同步；不能让部分线程提前覆盖共享存储。

若行宽 5、物理 block 宽 8，max 阶段的三个无效 lane 应使用 -∞，sum 阶段应贡献 0。若 max 阶段用 0，输入 [-5,-4,-3] 的最大值会被错误地变成 0；若 sum 阶段未屏蔽，概率和不再为 1。

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
python examples/cpu/softmax_lab.py
python -m unittest discover -s examples/cpu
~~~

将 block_size 改成 1、2、大于行长，比较稳定 reference 与 online 结果；测试全负、极大值、重复最大值、非 2 的幂行长。断言每个概率在 [0,1] 内、概率和接近 1、整体平移常数后输出不变。相对误差在接近 0 的位置不可靠，应同时看绝对误差。GPU 测试还要测行宽 31/32/33 和 1023/1024/1025，抓住末块边界。

性能实验记录行数、行宽、dtype、布局、kernel launch 数、总耗时和读写字节。极短行可能由 launch 或 block 利用率主导；极长行可能需要跨 block 分段。是否融合必须由数据决定。

## 5. 练习与答案要点

练习 A：对 [-5,-4,-3] 求稳定 softmax。减去 -3，得 [e^-2,e^-1,1] 再除以 1.503214，输出约 [0.090031,0.244728,0.665241]。练习 B：为什么 padding 的 max 单位元是 -∞？因为 max(x,-∞)=x；sum 的单位元是 0。练习 C：把 [1000,1001,999] 分成 [1000] 和 [1001,999] 两块，用上式合并，所得 m=1001、l≈1.503214。

## 参考资料（按学习顺序）

1. [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：观察一行到 block 的映射和 mask。
2. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：同步与内存模型。
3. [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：Online 统计在 Attention 中的延伸。
