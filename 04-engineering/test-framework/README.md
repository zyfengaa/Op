# 算子测试：不要只测一个随机输入

## 测试层次

1. 数学 reference：最清晰的 CPU/框架版本。
2. Shape matrix：小、中、大，规整和非规整。
3. Dtype matrix：支持类型和累加类型。
4. Layout matrix：contiguous 与 slice/transpose 形成的 non-contiguous。
5. Numeric matrix：极值、零、NaN/Inf、随机和病态输入。
6. Integration：device、stream、dispatch、autograd、compile 等目标功能。
7. Performance regression：固定硬件和输入的基线。

## 误差规则

使用绝对和相对容差，不用 bitwise equality。容差要符合 dtype、累加路径和算子语义。归约和矩阵乘法因加法顺序变化而产生细微差异很常见，但不能以此放宽到掩盖 bug。

## 边界测试生成

对于 tile T，至少覆盖 T-1、T、T+1；矩阵维度对 M、N、K 分别做非整除测试。还要覆盖空张量（若 API 支持）、单元素、极长维度和零 stride/广播语义（若支持）。

## 工具

CUDA 可用 Compute Sanitizer 检查内存与同步问题；PyTorch 自定义算子可用 opcheck 验证注册约定。opcheck 不等于数学正确性，也不自动证明梯度正确。

## 参考资料

- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [PyTorch opcheck](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html)
