# Kernel Fusion 学习实验：从语义等价到真实收益

先运行 [bias+ReLU CPU reference](../../assets/cpu/advanced_lab.py)。融合的第一关是结果相同，第二关才是收益；“少一个 kernel”不是充分的性能证明。

## 1. 具体输入和两种执行图

X=[[-2,1],[3,-4]]、bias=[1,2]，按列广播。T=X+bias=[[-1,3],[4,-2]]；Y=ReLU(T)=[[0,3],[4,0]]。分离实现先写 T 再读 T；融合版在计算 X+bias 的同一位置立即 ReLU 后写 Y。对 T 中 N 个 FP32 元素，中间读写理论上约 8N 字节。若 N=1,000,000，约 8 MB；这是算法层面估算，缓存和实际内存事务会改变 HBM 观测值。

## 2. 判断能否融合的四个条件

第一，语义一致：broadcast 轴、dtype 转换、舍入、NaN、原地/alias 规则不能悄然变化。第二，数据依赖允许：逐元素 bias+ReLU 很直接；若第二阶段要全局归约，就不是简单的单线程融合。第三，资源允许：融合增加活跃中间量，可能拉高寄存器占用和编译体积。第四，系统层面的收益：比较输入到输出的总耗时，不只挑最好看的单 kernel 时间。

例如 FP16 输入的分离路径若在第一阶段就舍入到 FP16，中间 T 被存储后再 ReLU；融合路径若一直用 FP32 算到最终输出，可能得到不同结果。若契约要求复现分离语义，就必须明确在哪里转换/舍入。又如 x 和输出共享内存时，写入顺序可能影响后续读取，不能凭公式自行允许 in-place。

## 3. 实验步骤和记录格式

~~~powershell
python assets/cpu/advanced_lab.py
python -m unittest discover -s assets/cpu
~~~

在 GPU 上实现 separate 与 fused 两版，选小 N、中 N、大 N，固定设备、频率环境、dtype 和布局。预热后重复计时并报告中位数或分布；记录 launch 数、kernel 时间、端到端时间、读写字节、寄存器/occupancy。若小 N 收益明显，可能主要来自 launch；若大 N 收益明显，可继续查实际内存流量；若融合变慢，检查资源压力和访存模式。实验表中明确区分预测流量与 profiler 实测流量。

## 4. 练习与答案

练习：对上面的 X/bias，融合结果是什么？答案 [[0,3],[4,0]]。若 T 为 4 个 FP32 元素，分离路径额外的 T 写与 T 读合计多少字节？4×4×2=32 字节。这个数字不包含 X/bias/Y 自身流量，也不等于设备实测 HBM 字节。

## 5. 三个阶段的流量账与中间精度

令 y=scale*relu(x+bias)，x 和 y 都有 N 个 FP32 元素。若 bias-add、ReLU、scale 分三次 kernel 执行，会有两个 N 元素中间张量。每个中间张量一次写出、一次读回，合计额外约 16N 字节。融合后两个中间值可在线程局部保存。

这只计算中间结果，不包含读取 bias、输入 x 和输出 y，也没计算 cache。若编译器已经融合了原路径，手写融合未必还能减少这些字节，所以第一步应查看实际执行的 kernel 数，而不是只数 Python 表达式中的操作符。

还有一个更难的正确性细节：如果原路径每一步都输出低精度 tensor，就在每个边界发生舍入；新路径若用更高精度保留全部中间值，结果可能不同。融合是否允许改变这些舍入点，必须由目标数值契约与测试决定。

## 6. 为什么 GEMM epilogue 比单独逐元素融合更有吸引力

GEMM 已在寄存器里持有 C 的累加结果。若接着要做 bias+activation，可在写回最终结果前完成，避免先把原始 C 写出，再由新 kernel 读入。这个尾部处理常称为 epilogue。

但 bias 的列索引必须跟输出 tile 对齐，尾块必须沿用合法输出 mask；若 GEMM 累加为 FP32、最终输出为 FP16，激活放在转换前还是转换后也改变数值路径。增加 epilogue 还可能延长 accumulator 的活跃期，影响寄存器和调度。

用前章 C=[[58,64],[139,154]]，bias=[-60,-150]，ReLU 后输出 [[0,0],[79,4]]。先手算它，再设计融合与未融合 reference 对照；如果第二列全错，先检查 bias 是否错误地按行广播。

## 7. 一个会增加全局依赖的“融合”反例

若下一阶段是对所有 C 求一个全局 sum，普通 block 不能只靠 __syncthreads 等待其他 block 的 C 完成。可以让各 block 先输出 partial sum，再开一个归约 kernel；这依然减少中间数据，但不是任意拼接两个函数就能变成安全的单 kernel。

如果采用 atomic 累加，需要正确初始化输出并讨论加法顺序；如果采用 cooperative launch，则要满足相应设备和启动条件。选择来自依赖关系，不能把“减少 launch”当成唯一标准。

## 8. 从单个输入扩展到调度策略

为 bias+ReLU 分别测 N 很小、N 中等、N 很大，以及连续与非连续输入。小输入的收益可能主要来自少一次 launch；大输入更关心带宽与资源；非连续输入若先做 contiguous copy，复制成本可能抵消融合。

如果只有连续大输入受益，可以让公共接口按条件分派至融合路径，其他输入使用 reference 或通用版本。测试需要覆盖每条分派分支，包括边界尺寸和不满足对齐的切片。优化版本的支持范围应成为可检查的条件，而不是文档里的一句提醒。

## 9. 练习：写一份能复现的融合报告

报告包含原始计算表达式、实际 kernel 序列、输入契约、理论中间流量、正确性误差、设备端时间、调用总时间和资源变化。对每条性能解释写出它对应的证据：少一次 launch 看时间线，少显存读取看计数器，寄存器上升看编译/profiler 信息。

若结果变慢，保留这个失败实验和适用输入。它能说明哪种条件下简单融合不合适，比只展示一个最好看的 shape 更能训练判断力。

## 参考资料

1. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：性能测量与内存访问。
2. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)：查看融合 kernel 案例。
3. [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：接口、注册与语义验证。
