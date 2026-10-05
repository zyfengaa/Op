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

## 从一个真实失败反推测试，而不只是罗列 shape

假设 GEMM 在 32×32×32 上通过，37×29×41 上失败。先分别只让 M、N、K 非整除：只有 K 非整除失败通常提示 tile 补零/循环轮数；只有 M/N 尾部失败提示输出或装载 mask。把所有维度一起改虽可发现 bug，却不利于定位。

对归约，选择长度为线程数-1、线程数、线程数+1，再选择远大于线程数的长度；前者检查空 lane 和尾部，后者检查每线程多元素循环。对 LayerNorm，加一组常量行与不同 gamma/beta，检查统计和广播两个环节。

## 有限浮点数的误差判断怎样写

常用判断为 abs(actual-reference)<=atol+rtol*abs(reference)。atol 负责参考值接近 0 的区域，rtol 随结果大小放缩。先明确 dtype 和累加路径再选择阈值，并在失败时打印最大误差的位置，而不是只有 FAIL。

还有一个容易忽视的测试 bug：

~~~cpp
if (std::abs(actual - expected) > tolerance) fail();
~~~

如果 actual 是 NaN，比较表达式可能为 false，测试反而通过。对本课程定义为有限输出的案例，应先检查 std::isfinite(actual)，再做容差比较。本仓库已有 CUDA 示例和新 row_ops 都补了这项检查。若算子契约确实允许 NaN/Inf，则应单独验证对应位置的特值语义，而不是一律拒绝。

整数输入的小测试适合验证索引，但容易掩盖浮点累计误差。因此还要测试具有不同数量级的非整数输入。不要用增加 tolerance 来修复明显的整行错位或末块错误。

## reference 必须尽量独立

用同一段 kernel 代码换一个 block size 当作 reference，可能让两边共享同一个索引错误。GEMM 的朴素 CPU 三重循环与 GPU 分块循环分工不同，更容易发现 tile 错误；显式 attention 与 online m/l/o 合并是不同计算路径，可以互相核验。

本仓库 [test_dataflow_lab.py](../assets/cpu/test_dataflow_lab.py) 检查 Welford 不同分块的合并、在线 Attention 对显式 reference、im2col 对直接卷积，以及明确的手算中间状态。它们验证算法，不验证设备同步。CUDA 的越界、共享竞争和框架 stream 行为仍需设备测试。

## 不依赖固定答案的性质测试

Softmax 在有限合理范围内应对整行共同平移保持不变；其概率和接近 1。LayerNorm 常量行在正 eps 下输出 beta。Attention 无 dropout 且权重有效时，每个输出分量是可见 V 分量的凸组合。GEMM 满足与零矩阵相乘为零，以及乘单位矩阵保持输入。

性质测试能扩大检查范围，但不是充分证明。例如错误的均匀 softmax 也满足和为 1，必须再与 reference 比较。每条性质都要写清前提，尤其不能忽略浮点容差和特殊值。

## 本仓库的三个执行入口

~~~powershell
python -m unittest discover -s assets/cpu -v
python 05-frameworks/impl/custom_bias_relu.py
ctest --test-dir build/cuda -C Release --output-on-failure
~~~

第一项只需标准 Python；第二项需要兼容 PyTorch，CPU 可运行；第三项要求先构建 CUDA 可执行程序并有可运行的 GPU 环境。结果要分别记录，不把 CPU PASS 汇总成“所有后端通过”。

## 失败日志应保留什么

至少记录 commit、shape、dtype、stride、seed、失败坐标、实际/期望值、误差阈值、设备与软件版本。随机失败应保存能复现的 seed 或输入。并发相关失败再附上 stream/launch 配置及 sanitizer 日志。

若首次失败发生在同步或复制处，继续追查此前的 kernel；那可能只是异步错误被报告的位置。缩小 shape、只保留一个 kernel、用结构化输入定位，通常比反复修改整个实现更有效。

## 参考资料

- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [PyTorch opcheck](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html)
