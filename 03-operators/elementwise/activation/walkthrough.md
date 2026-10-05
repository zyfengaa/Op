# 逐元素激活：把一个公式变成可信的算子

读完本章，你应能解释为什么逐元素计算不需要线程间同步、写出尾部保护、区分不同 GELU 定义，并用边界输入定位数值问题。先运行 [CPU 实验](../../../assets/cpu/elementwise_lab.py)，再把每个独立元素映射到 GPU 线程。

## 1. 先确定契约，而不是先写 kernel

输入为长度 N 的一维数组 x，输出 y 与 x 同形。ReLU 是 y[i]=max(x[i],0)，但当 x[i] 是 NaN 时，语言和框架对 max 的处理可能不同，不能只看代数式。Sigmoid 是 1/(1+exp(-x))。GELU 有 erf 精确形式和 tanh 近似形式，它们不是同一个契约。先固定输入 dtype、计算 dtype、输出 dtype、是否支持非连续布局、NaN/Inf 语义和容差。

手算例子：x=[-2,0,3]，ReLU(x)=[0,0,3]。若再乘 2，输出 [0,0,6]。这里三个输出互不依赖，所以完全可以并行。

## 2. 从 CPU 循环到 GPU 线程

CPU reference 的核心是对每个 i 计算 y[i]=f(x[i])。GPU 中令 i=blockIdx.x*blockDim.x+threadIdx.x，只有 i<N 才读写。N=10、blockDim=4 时会发出 12 个线程，最后两个线程必须退出。对二维非连续张量，线性 i 不能直接当作物理地址：应先从 i 求逻辑坐标，再按 stride 计算地址；否则 transpose/slice 输入会悄悄算错。

每个有效线程通常执行“读一次、算一次、写一次”。这是独立元素模型，无需 shared memory，也无需 __syncthreads。加入同步不但没有用，还可能在分支中造成死锁。

## 3. 数值推导：为什么 Sigmoid 要分支

对 x=-1000，直接计算 exp(-x)=exp(1000) 可能溢出。稳定实现分两支：

~~~text
x >= 0: z = exp(-x); sigmoid = 1/(1+z)
x <  0: z = exp(x);  sigmoid = z/(1+z)
~~~

第二支对很负的 x 先得到接近 0 的 z，不需要生成巨大指数。用 CPU 实验确认 x=1000 与 -1000 的结果分别接近 1 和 0。注意“稳定”不意味着所有中间 dtype 都一样：FP16 输入可转换成 FP32 计算，再按接口要求转回 FP16。

GELU 精确形式为 0.5*x*(1+erf(x/sqrt(2)))；tanh 形式是近似。比较两版时先用同一个公式做 reference，否则差异不说明 kernel 错。对大负输入，结果接近 0；对大正输入，结果接近 x。

## 4. 能运行的实验

在仓库根目录运行：

~~~powershell
python assets/cpu/elementwise_lab.py
python -m unittest discover -s assets/cpu
~~~

先预测打印的 ReLU、Sigmoid 与 GELU，再看输出。自己增加 [-1000,-10,-1,0,1,10,1000]，检查单调性 sigmoid(-x)=1-sigmoid(x) 在合理误差内成立。试验 NaN 时，只与目标框架约定比较，不能把 Python 行为直接当成所有 GPU 实现的标准。

## 5. 融合：减少什么，可能失去什么

分离的 bias 与 ReLU 需要：读 x/bias，写临时 t；再读 t，写 y。融合后一次读取 x/bias、一次写 y。假设 x 有 N 个 FP32 元素，仅临时 t 的写回与重读就约为 8N 字节；这是流量上界估算，不是实测收益。小张量往往还会减少一次 launch；大张量可能因寄存器、带宽和调度变化而不按估算加速。

动手把 y=ReLU(x+bias) 的 CPU 版写成“分离”和“融合”两版，测试结果一致。GPU 版再记录端到端时间与 kernel 时间，不能只报单个融合 kernel 更快。

## 6. 排错清单与练习

- 末块错误：用 N=blockDim-1、blockDim、blockDim+1 测。
- 布局错误：用转置或切片形成 stride 不等于 1 的输入测。
- 公式错误：明确 GELU 是 erf 还是 tanh。
- 精度错误：分别比较 FP32、FP16/BF16 输入的容差。
- 特值错误：NaN/Inf 对照目标框架而非凭直觉。
- 性能误判：记录 shape、dtype、设备、warmup、重复次数。

练习 A：手算 x=[-1,0,2] 的 ReLU 与 3*ReLU。答案分别为 [0,0,2] 与 [0,0,6]。练习 B：N=9、blockDim=4 需要多少 block？答案为 3，末 block 只有一个有效线程。练习 C：为什么 sigmoid(-1000) 不用 exp(1000)？答案是采用 exp(x)/(1+exp(x)) 的负数分支，避免正指数溢出。

## 7. 输入、计算与输出 dtype 可以不同

FP16/BF16 存储减少字节数，但它们的范围与精度取舍不同。一个常见计算路径是先将输入值转换到 FP32，做指数/近似函数计算，再转换成目标输出 dtype。这能改善中间运算，却无法恢复输入存储时已经损失的信息。

因此精度测试应有两份对照：一份使用相同的已量化输入值评估 kernel 计算误差，另一份在需要时比较整个低精度路径相对原始高精度输入的误差。否则可能把输入量化归罪于 kernel，或者把中间计算错误隐藏在总误差里。

不要从“FP16 字节数减半”直接推出速度翻倍。Sigmoid/GELU 有指数、erf/tanh 等运算，可能受到数学指令吞吐、转换或指令依赖影响；小输入还可能主要受 launch 影响。

## 8. GELU 为什么会输出负数

ReLU 把所有负值截成 0；精确 GELU 是 x*Φ(x)，Φ 是标准正态分布累计概率，对有限负 x，Φ(x) 仍是正数，所以结果通常是小的负数。例如 x=-1 时，Φ(-1)≈0.158655，GELU(-1)≈-0.158655。若“修复”成 max(GELU(x),0)，就改变了算子。

tanh 近似和 erf 形式在有限区间接近，但不是位级一致。实验时明确 reference 的 approximate 参数，再比较误差；只有两边采用同一种数学契约，误差才能说明实现质量。

## 9. 广播与非连续输入如何落到地址

对 x[B,D]+bias[D]，线性输出 i 对应 row=i//D、col=i%D，bias 地址应是 bias[col]。如果写 bias[i]，第二行开始就越界；如果写 bias[row]，在 B=D 的方阵测试中可能不越界，却语义错误。

对非连续 x，地址需使用 row*stride0+col*stride1，而不能无条件用 i。输出可以新分配连续，也可以按接口支持 out stride；这两种策略的约束不同。先在 [2,3] 等非方阵上画地址表，再用转置视图验证，容易发现把 row/col 混淆的问题。

## 10. 激活反向是一次数学与接口的联合练习

ReLU 的反向在正输入处传递梯度，在负输入处置零；0 处采用目标框架约定。Sigmoid 若保存前向输出 s，导数可写 s*(1-s)；这省去重新算指数，但增加保存状态。GELU 的不同近似形式也有不同导数，不能前向用 tanh 近似、反向却未经说明用另一公式。

本仓库 [PyTorch bias+ReLU 示例](../../../05-frameworks/impl/custom_bias_relu.py) 可用 CPU 检查 forward、broadcast 梯度和注册。数值 gradcheck 应避开 ReLU 的不可微点；如果差分两侧跨过 0，失败不必然说明 backward 代码错。

## 参考资料（按学习顺序）

1. [CUDA 编程模型：线程和 block](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。
2. [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)：核对数学函数及精度。
3. [PyTorch GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)：核对近似参数语义。
4. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存与性能测量。
