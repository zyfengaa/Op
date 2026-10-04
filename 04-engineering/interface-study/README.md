# 接口层、设备层和 kernel 层

## 建议分层

~~~text
Public API / schema
    ↓ 参数检查、dispatch、错误信息
Backend adapter
    ↓ 设备、stream、workspace、runtime
Kernel implementation
    ↓ CUDA / Triton / vendor language
Tests and reference
    ↓ backend-independent semantics
~~~

公共 API 定义行为；backend adapter 处理设备特定资源；kernel 层只实现计算。不要让 CUDA 类型泄漏到所有公共接口，也不要为了抽象而隐藏必要设备约束。

## 接口检查项

- shape、stride、dtype、device。
- layout 和对齐要求。
- 输出分配或 out 参数。
- workspace 大小与生命周期。
- stream/queue 和同步规则。
- 原地修改、alias 和错误处理。
- 支持的精度和近似公式。

## 多 backend 的组织

公共层放数学语义、reference、误差阈值、输入生成和性能接口；backend 目录隔离编译、runtime、同步、profiling 和专用指令。warp 宽度、向量宽度、bank 数量不得写成全局假设。

## 参考资料

- [PyTorch dispatcher and custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch PrivateUse1 backend integration](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
