# Conv2D 学习实验：输出坐标、布局与算法选择

本章使用 [单通道 CPU reference](../../assets/cpu/advanced_lab.py) 先搞清楚索引。它只覆盖单通道、dilation=1、groups=1 的互相关；完整框架算子还有 batch、通道、groups、bias 和不同布局，不能把这个小例误当成完整 Conv2D。

## 1. 先手算一个输出

输入 X 为 3×3 矩阵 [[1,2,3],[4,5,6],[7,8,9]]，filter 为 [[1,0],[0,-1]]，stride=1、pad=0。输出高宽各为 3-2+1=2。左上角是 1×1+2×0+4×0+5×(-1)=-4；四个窗口全部得到 -4，所以输出 [[-4,-4],[-4,-4]]。这里 filter 没有翻转，因此是深度学习框架常用的互相关约定。若先翻转 filter，结果会不同。

一般输出高度为 floor((H+2P_h-D_h(K_h-1)-1)/S_h)+1，宽度同理。每个输出位置先由 stride 定位窗口左上角，再扣去 padding 偏移；越界输入贡献 0。pad=1 时，即使输入坐标为负，输出位置仍可能合法。

## 2. 多通道与布局如何扩展

对 NCHW，逻辑元素 (n,c,h,w) 的连续地址是 ((n*C+c)*H+h)*W+w；NHWC 的地址是 ((n*H+h)*W+w)*C+c。对于带任意 stride 的张量，必须使用实际 stride 计算，不能套连续公式。每个输出 (n,co,oh,ow) 需要遍历属于该 group 的输入通道 ci，以及 filter 高宽 ky/kx。若有 bias，累加结束后再加 bias[co]。groups=C 且输出通道匹配时是 depthwise 的特例，和普通卷积的通道访问不同。

## 3. 直接卷积、im2col、Winograd 怎么比

直接卷积不制造巨大的展开矩阵，但重复窗口读取可能多；cache 和 tile 可以改善复用。im2col 将每个窗口展成一行，随后做 GEMM；若有大量窗口，展开空间约为 N*OH*OW*(CI*KH*KW) 个元素。它能复用成熟 GEMM，但准备矩阵的时间和内存必须算进端到端成本。Winograd 在特定小 filter/shape 上减少乘法，代价是变换、额外数值误差和适用限制；不能只数乘法就宣布更快。

## 4. 运行与测试

~~~powershell
python assets/cpu/advanced_lab.py
python -m unittest discover -s assets/cpu
~~~

先修改 stride=2、pad=1，画出每个输出窗口，逐项手算。然后测试输入小于 filter、非方阵、非对称 filter，抓住输出 shape 和 filter 翻转错误。扩展到 GPU 时，再测试 NCHW/NHWC、groups、dilation、bias、不同 dtype；与 PyTorch 对比前必须匹配所有参数。性能实验要报告展开时间、GEMM 时间、临时内存与整体耗时。

### 4.1 练习与答案

输入 3×3、filter 2×2、stride=2、pad=0，输出尺寸是多少？floor((3-2)/2)+1=1，因此 1×1。上述 [[1,0],[0,-1]] filter 对左上角窗口输出 -4。

## 5. 真正把四个窗口展开成矩阵

对前面的 3×3 输入与 2×2 filter，窗口按照输出行优先展开。每个窗口内部按 ky/kx 行优先展开：

~~~text
W = [[1,2,4,5],
     [2,3,5,6],
     [4,5,7,8],
     [5,6,8,9]]

F = [[ 1],
     [ 0],
     [ 0],
     [-1]]
~~~

W 形状为 [OH*OW,KH*KW]=[4,4]，F 形状为 [4,1]。W@F 得到四个 -4，按 [OH,OW]=[2,2] reshape 回输出。直接卷积和这条路径计算同一公式，但数据组织不同。

运行 [im2col 实验](../../assets/cpu/dataflow_lab.py)：

~~~powershell
python assets/cpu/dataflow_lab.py --topic conv
~~~

程序会打印上述 W 和最后结果。它明确生成 W 再调用 CPU GEMM，所以确实包含展开步骤；不能把这段程序的耗时当作 GPU im2col 性能。测试采用不同形状和非对称 filter 与直接卷积 reference 对照，防止一个“输出全相同”的例子掩盖索引错误。

## 6. 推导通用矩阵形状

若输入通道数为 CI、输出通道数为 CO、groups=1，每个窗口含 CI*KH*KW 个数。令 R=N*OH*OW、K=CI*KH*KW，im2col 矩阵为 [R,K]，重排后的 filter 为 [K,CO]，输出矩阵为 [R,CO]。

例如 N=1、CI=2、CO=3、H=W=4、KH=KW=3、stride=1、pad=0，则 OH=OW=2，R=4，K=18，最终 GEMM 是 [4,18]@[18,3]。输出 12 个元素要再按所需 NCHW 或 NHWC 布局解释/重排，不能只看到 GEMM 结果 shape 就结束。

展开矩阵含 R*K 个元素；这个例子为 72 个 FP32 数，288 字节。输入原本只有 32 个数，128 字节。窗口重叠造成数据复制，随着参数变化，展开开销可能很大。这些是形状演算，不是内存 profiler 结果。

## 7. “隐式 GEMM”省掉了什么，又增加了什么

显式 im2col 把所有窗口先写入全局内存，再让 GEMM 读取。隐式 GEMM 在准备每个计算 tile 时，根据矩阵坐标反推出原输入的 n/c/h/w，只把当前需要的数据搬进片上缓冲，避免完整展开矩阵落盘。

代价是装载地址计算更复杂，还要处理 padding、dilation 和 groups。输入坐标不连续时，加载向量化也更困难。理解这一点，就能把卷积优化与前面的 GEMM tiling 联系起来：数学上像 GEMM，输入数据的获取方式却不同。

## 8. groups 和 dilation 的具体索引

分组卷积中，每个输出通道只连接本组的 CI/groups 个输入通道。例如 CI=4、CO=4、groups=2，输出通道 0/1 只看输入通道 0/1，输出通道 2/3 只看输入通道 2/3。若仍累加全部 CI，结果语义就变了。

dilation=2、KH=3 时，filter 三行覆盖的输入偏移为 0、2、4，有效覆盖高度为 2*(3-1)+1=5，而不是 3。输出形状公式中的 D*(K-1)+1 正是这个覆盖范围。分别改变 dilation 和 stride，观察一个改变窗口内部间距，另一个改变相邻窗口起点间距。

## 9. Winograd 的学习边界

Winograd 在适用的小卷积条件下，通过输入/权重变换、逐元素乘法和输出逆变换减少乘法数量。它并不消灭计算，而是改变乘法与加法/变换的分配；低精度下误差也会受变换影响。

学习时先能比较直接卷积和显式 im2col 的数值与存储，再读实际库如何按 shape/dtype/工作空间选择算法。不要为所有 3×3 filter 默认指定同一算法，尤其不能忽略 workspace 与端到端变换成本。

## 10. 有目的的测试输入

使用单个非零像素观察它能影响哪些输出，能有效定位 padding/stride 错误；使用只在 filter 左上角非零的权重定位翻转与坐标偏移；让各通道填入不同常数定位 groups；用 H≠W、KH≠KW 定位高宽混用。

这些输入比完全随机数据更容易解释失败原因。随机测试负责扩大覆盖，结构化输入负责帮助定位，两者都需要。

## 参考资料

1. [PyTorch Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)：检查参数和输出形状约定。
2. [cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/)：了解库算法选择与布局。
3. [CUTLASS Convolution](https://docs.nvidia.com/cutlass/)：进阶隐式 GEMM 与 tile。
