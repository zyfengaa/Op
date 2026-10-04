# Conv2D 学习实验：输出坐标、布局与算法选择

本章使用 [单通道 CPU reference](../../examples/cpu/advanced_lab.py) 先搞清楚索引。它只覆盖单通道、dilation=1、groups=1 的互相关；完整框架算子还有 batch、通道、groups、bias 和不同布局，不能把这个小例误当成完整 Conv2D。

## 1. 先手算一个输出

输入 X 为 3×3 矩阵 [[1,2,3],[4,5,6],[7,8,9]]，filter 为 [[1,0],[0,-1]]，stride=1、pad=0。输出高宽各为 3-2+1=2。左上角是 1×1+2×0+4×0+5×(-1)=-4；四个窗口全部得到 -4，所以输出 [[-4,-4],[-4,-4]]。这里 filter 没有翻转，因此是深度学习框架常用的互相关约定。若先翻转 filter，结果会不同。

一般输出高度为 floor((H+2P_h-D_h(K_h-1)-1)/S_h)+1，宽度同理。每个输出位置先由 stride 定位窗口左上角，再扣去 padding 偏移；越界输入贡献 0。pad=1 时，即使输入坐标为负，输出位置仍可能合法。

## 2. 多通道与布局如何扩展

对 NCHW，逻辑元素 (n,c,h,w) 的连续地址是 ((n*C+c)*H+h)*W+w；NHWC 的地址是 ((n*H+h)*W+w)*C+c。对于带任意 stride 的张量，必须使用实际 stride 计算，不能套连续公式。每个输出 (n,co,oh,ow) 需要遍历属于该 group 的输入通道 ci，以及 filter 高宽 ky/kx。若有 bias，累加结束后再加 bias[co]。groups=C 且输出通道匹配时是 depthwise 的特例，和普通卷积的通道访问不同。

## 3. 直接卷积、im2col、Winograd 怎么比

直接卷积不制造巨大的展开矩阵，但重复窗口读取可能多；cache 和 tile 可以改善复用。im2col 将每个窗口展成一行，随后做 GEMM；若有大量窗口，展开空间约为 N*OH*OW*(CI*KH*KW) 个元素。它能复用成熟 GEMM，但准备矩阵的时间和内存必须算进端到端成本。Winograd 在特定小 filter/shape 上减少乘法，代价是变换、额外数值误差和适用限制；不能只数乘法就宣布更快。

## 4. 运行与测试

~~~powershell
python examples/cpu/advanced_lab.py
python -m unittest discover -s examples/cpu
~~~

先修改 stride=2、pad=1，画出每个输出窗口，逐项手算。然后测试输入小于 filter、非方阵、非对称 filter，抓住输出 shape 和 filter 翻转错误。扩展到 GPU 时，再测试 NCHW/NHWC、groups、dilation、bias、不同 dtype；与 PyTorch 对比前必须匹配所有参数。性能实验要报告展开时间、GEMM 时间、临时内存与整体耗时。

## 练习与答案

输入 3×3、filter 2×2、stride=2、pad=0，输出尺寸是多少？floor((3-2)/2)+1=1，因此 1×1。上述 [[1,0],[0,-1]] filter 对左上角窗口输出 -4。

## 参考资料

1. [PyTorch Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)：检查参数和输出形状约定。
2. [cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/)：了解库算法选择与布局。
3. [CUTLASS Convolution](https://docs.nvidia.com/cutlass/)：进阶隐式 GEMM 与 tile。
