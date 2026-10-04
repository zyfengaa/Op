# Conv2D：算法变换和数据布局权衡

先读 [Conv2D 逐步实验](walkthrough.md)，先手算窗口与地址，再比较直接卷积和 im2col。

## 语义先写清楚

定义 NCHW/NHWC、stride、padding、dilation、groups、bias 和输出 shape。卷积框架常采用互相关定义，即 filter 不翻转；测试必须与目标框架约定一致。

## 三条实现路线

### 直接卷积

每个输出位置累加输入窗口与 filter 的乘积。优点是没有大型临时矩阵；缺点是访存复用和矩阵单元利用率可能不足。

### im2col + GEMM

把滑动窗口展开成矩阵，再调用 GEMM。优点是复用成熟 GEMM；缺点是展开产生额外存储和 HBM 流量，可能使小卷积变慢。

### Winograd

对特定小 filter（常见 3×3）减少乘法次数，但有变换开销、数值误差和 shape 限制。乘法减少不自动等于端到端更快。

## 小实验

固定卷积参数，分别实现 CPU reference、直接卷积和 im2col+GEMM；测不同 batch、通道数、空间尺寸和布局。记录临时内存、变换成本、kernel 时间和总时间。

## 常见错误

- 输出 shape 的 padding/stride 公式写错。
- NCHW 与 NHWC 地址公式混用。
- groups>1 时仍按普通卷积处理 channel。
- 只测大 shape，忽略小 batch 下 im2col 成本。
- 忽略低精度算法的误差增长。

## 参考资料

- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUTLASS Convolution](https://docs.nvidia.com/cutlass/)
- [PyTorch Conv2d semantics](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
