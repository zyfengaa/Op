# Elementwise Activation：ReLU、Sigmoid、GELU

## 本节目标

从 Vector Add 迁移到逐元素数学函数，学会验证边界、dtype、NaN/Inf 和融合的收益。

## 三个算子定义

~~~text
ReLU(x) = max(x, 0)
Sigmoid(x) = 1 / (1 + exp(-x))
GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
~~~

注意 GELU 有精确 erf 形式和常用 tanh 近似形式；接口与测试必须明确采用哪一个，否则“误差不一致”不一定是 kernel bug。

## 从公式到 kernel

逐元素 activation 沿用 `i = blockIdx.x*blockDim.x+threadIdx.x`。每个有效线程读取 X[i]、计算 f(x)、写 Y[i]。Elementwise 通常没有线程间依赖，所以不需要 shared memory 或 block barrier。

## 数值测试矩阵

- 普通随机值、0、正负极值。
- Sigmoid 的大正数和大负数，检查溢出/下溢是否符合预期。
- NaN、+Inf、-Inf，确认与框架 reference 的语义一致。
- FP32 与 FP16/BF16，确认输入、计算和输出 dtype 规则。

## 融合示例

~~~text
两个 kernel: tmp = relu(x); y = scale(tmp)
一个融合 kernel: y = scale(relu(x))
~~~

融合可减少 launch 和中间 tensor 的写回/重读，但可能增加寄存器占用或代码复杂度。要比较完整路径，不只看单个 kernel 指标。

## 常见错误

- GELU 近似公式与 reference 使用不同形式。
- 对半精度直接使用过低精度中间运算，造成误差积累。
- 通过融合减少流量，却漏掉输入可能 alias 的框架约束。
- 以为 elementwise 一定带宽打满；小张量可能受 launch bound 限制。

## 练习与参考

实现 ReLU 和 Sigmoid，与 PyTorch/CPU reference 比较；把 `bias + GELU` 融合并测量中间 tensor 字节数和 E2E 时间。

- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
- [PyTorch GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
