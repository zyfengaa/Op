# Kernel Fusion 学习实验：从语义等价到真实收益

先运行 [bias+ReLU CPU reference](../../examples/cpu/advanced_lab.py)。融合的第一关是结果相同，第二关才是收益；“少一个 kernel”不是充分的性能证明。

## 1. 具体输入和两种执行图

X=[[-2,1],[3,-4]]、bias=[1,2]，按列广播。T=X+bias=[[-1,3],[4,-2]]；Y=ReLU(T)=[[0,3],[4,0]]。分离实现先写 T 再读 T；融合版在计算 X+bias 的同一位置立即 ReLU 后写 Y。对 T 中 N 个 FP32 元素，中间读写理论上约 8N 字节。若 N=1,000,000，约 8 MB；这是算法层面估算，缓存和实际内存事务会改变 HBM 观测值。

## 2. 判断能否融合的四个条件

第一，语义一致：broadcast 轴、dtype 转换、舍入、NaN、原地/alias 规则不能悄然变化。第二，数据依赖允许：逐元素 bias+ReLU 很直接；若第二阶段要全局归约，就不是简单的单线程融合。第三，资源允许：融合增加活跃中间量，可能拉高寄存器占用和编译体积。第四，系统层面的收益：比较输入到输出的总耗时，不只挑最好看的单 kernel 时间。

例如 FP16 输入的分离路径若在第一阶段就舍入到 FP16，中间 T 被存储后再 ReLU；融合路径若一直用 FP32 算到最终输出，可能得到不同结果。若契约要求复现分离语义，就必须明确在哪里转换/舍入。又如 x 和输出共享内存时，写入顺序可能影响后续读取，不能凭公式自行允许 in-place。

## 3. 实验步骤和记录格式

~~~powershell
python examples/cpu/advanced_lab.py
python -m unittest discover -s examples/cpu
~~~

在 GPU 上实现 separate 与 fused 两版，选小 N、中 N、大 N，固定设备、频率环境、dtype 和布局。预热后重复计时并报告中位数或分布；记录 launch 数、kernel 时间、端到端时间、读写字节、寄存器/occupancy。若小 N 收益明显，可能主要来自 launch；若大 N 收益明显，可继续查实际内存流量；若融合变慢，检查资源压力和访存模式。实验表中明确区分预测流量与 profiler 实测流量。

## 4. 练习与答案

练习：对上面的 X/bias，融合结果是什么？答案 [[0,3],[4,0]]。若 T 为 4 个 FP32 元素，分离路径额外的 T 写与 T 读合计多少字节？4×4×2=32 字节。这个数字不包含 X/bias/Y 自身流量，也不等于设备实测 HBM 字节。

## 参考资料

1. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：性能测量与内存访问。
2. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)：查看融合 kernel 案例。
3. [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：接口、注册与语义验证。
