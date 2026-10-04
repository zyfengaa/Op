# 完整算子交付：以 LayerNorm 为例

这章回答“算对了以后还差什么”。假设团队要交付一个前向 LayerNorm 算子，目标是支持二维连续 FP32 输入 [B,D]，按最后一维归一化，gamma/beta 长度 D，eps>0。不支持非连续输入、FP16 或反向传播时必须明确报错或回退，不能静默给错误结果。

## 1. 写规格：让实现和测试使用同一份契约

建议在开工前写下：输入 shape、dtype、device、stride；输出 shape、dtype、是否允许 alias；数学公式及方差除数；eps 位置；gamma/beta 广播；空维度策略；误差阈值；是否要求确定性。对本例，[2,4] 的 x、[4] 的 gamma/beta 应返回 [2,4]。gamma 长度 3 应给清晰错误。非连续输入若不支持，应检测后报错或显式 contiguous 并记录代价，不能直接假定物理行宽 D。

在公共层保存这份规格和清晰 CPU reference；后端只负责在特定设备上实现同样语义。这样未来增加 Triton 或其他平台时，不用重新发明测试定义。

## 2. 第一条纵向切片：先有可验证的最小实现

先选 B=1、D=4；用 [LayerNorm 手算例](../02-perf-aware-ops/02-layernorm/walkthrough.md) 的 [1,2,3,4] 对照。写 host wrapper 做参数检查和输出分配，调用最朴素 GPU kernel。与 reference 比较输出和统计量。暂不优化，也不要把计时结果当成最终性能。

接着测 D=1、D=tile-1、tile、tile+1、B>1、常量行、大动态范围、gamma/beta 不全为 1/0。若支持 FP16，需额外决定累加类型和容差；若不支持，加入“拒绝 FP16”的测试。

## 3. 接口/后端/内核的职责

~~~text
public API: 语义检查、shape/dtype/layout、错误信息
backend adapter: device、stream、workspace、launch 配置
kernel: 每个线程处理哪些元素、归约/同步/写回
reference/tests: 独立判断结果是否正确
benchmark: 固定输入与环境，记录可复现实验
~~~

kernel 内不宜随意分配宿主内存；wrapper 不应偷偷同步整个设备以“保证正确”，因为这会改变异步语义和测量结果。异步设备错误通常在后续同步点暴露，调试时要区分真正出错的 launch 与报告错误的位置。输出 buffer 的生命周期和 stream 使用都要可解释。

## 4. 测试分层：每一层证明什么

单元测试验证公式：常量行输出 beta、手算输入符合预期。参数测试验证错误路径：D=0、gamma/beta 形状不匹配、错误 dtype/设备。后端测试验证索引与同步：tile±1、多个 block、随机与极值输入。工具检查验证越界和竞争；它们不能代替数值比较。框架集成测试验证注册、dispatch、shape 推断以及目标场景需要的 autograd/compile；这些也不能代替 kernel 正确性。性能回归测试验证选定设备上没有显著退化，但普通共享 CI runner 的波动不能当成微小性能差异。

在无 GPU 机器上至少运行：

~~~powershell
python -m unittest discover -s examples/cpu -v
~~~

这是算法 reference 测试，不等于 GPU kernel 已经通过。CUDA 可用环境应另跑构建、设备测试和 sanitizer，并保存原始日志。

## 5. 性能实验的最小记录

每条实验写下：Git commit 与 dirty 状态、设备/驱动/编译器、shape、dtype、layout、输入分布、warmup 次数、测量次数、统计量、kernel 与端到端时间、reference 误差、profiler 证据、是否保留改动。示例表头：

| 版本 | B×D | dtype | 正确性 | kernel 中位数 | E2E 中位数 | 主要证据 |
|---|---:|---|---|---|---|---|
| baseline | 128×1024 | FP32 | 通过 | 待实测 | 待实测 | 两遍归约 |
| fused | 128×1024 | FP32 | 待测 | 待实测 | 待实测 | 待 profile |

“待实测”不能换成虚构数字。提速比只对同设备、同输入、同测量方法有意义；缩短单个 kernel 未必降低总耗时。

## 6. 交付审查和练习

PR/提交至少包含规格、reference、实现、边界测试、运行说明和限制。审查者应能回答：输入非连续怎么办？D 不整除 block 怎么办？方差除数是什么？eps 放在哪？FP16 在哪一步转换？错误如何报告？性能是 kernel 还是端到端？当前设备上是否实测？

练习：为上述 LayerNorm 写三条拒绝路径。参考答案：gamma 长度 !=D；输入 dtype 不支持；非连续输入且接口声明仅支持连续布局。每条都应有稳定、可理解的错误信息。

## 参考资料（按用途）

1. [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：注册、测试与框架集成。
2. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：验证与测量。
3. [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：越界和竞争排查。
4. [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)：目标架构与构建边界。
