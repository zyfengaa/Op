# 让 decode 小 M GEMM 回到真实模型位置

旧版 [decode GEMM 分析](../archive-legacy-docs/01_decode_gemm_performance_analysis.md) 使用模拟 GPU 数字训练分析方法。本章保留那条推理路线，但从可以运行的 block、实际 shape 和本机报告开始，不把旧模拟百分比移入新的实测表。

## 从投影调用找到 M/K/N

在 block.py 中，normed@[D,3D] 对应 packed QKV；MLP 的 normed@[D,2F] 对应 gate/up。前缀处理时 M=B*S；单 token decode 时 M=B。在默认 B=1 下，QKV 由 M=64 变成 M=1，而 K=64、N=192 保持不变。

运行 run.py 后，在 summary.json 的 shapes 查看这几组实际参数。不要看到 profiler 里的 aten::mm 就断言它是 Attention 的 QKᵀ；它可能是 QKV、O、gate/up 或 down 投影。slice/ 标注提供了模型层级上下文。

## 为什么权重复用不同

一次 Y=XW 的近似计算量是 2MNK。只按一遍权重读取估算，字节数为 KN*s，s 是每个权重的字节数，因此算术强度约为 2M/s。M=1、FP32 时约 0.5 FLOP/byte；FP16 时约 1。增大 M 会让同一个权重服务更多输入行。

这是忽略 cache、输入输出和实现重复访问的模型。当前 D=64 的权重很小，可能在 CPU cache 中；不能据这个公式直接认定本机 tiny block 是 HBM memory-bound。旧示例的大维度 GPU 推论要在相应设备和工作集下重新验证。

## 一个由运行结果触发的诊断练习

当前 CPU 报告中 optimized decode 的 instrumented 阶段占比最高的是 RoPE，约占这些 CPU stage 范围总和的 33.09%。这说明在这个教学实现和输入下，创建位置、三角函数和多次小 tensor 操作值得检查；它不能推出大型 CUDA decode 的热点也是 RoPE。

尝试预计算 cos/sin 表时，先保证读取的绝对位置和 dtype 一致，再测完整 block。若只记录 QKV 微基准而没有重跑 E2E，就无法知道实际请求是否受益。

当前候选把 QKV/gate-up 投影合并，还切换了 SDPA，因此需要消融实验才能归因。拆成三个配置：只合并 QKV、只合并 gate/up、只切换 Attention，再加组合版。保存结果，即使某个版本更慢。

## 改 batch、上下文与宽度时分别发生什么

- batch 改变 decode GEMM 的 M，也改变同一调用中服务的 token 数。
- prefix length 改变 Attention 读取的 K/V 长度，通常不改变投影矩阵的 K/N。
- width 改变投影的 K/N 和 head_dim（若 heads 不变），影响矩阵工作量与权重大小。
- hidden 改变 MLP gate/up/down 的形状，不直接改变 QKV 的形状。

不要一次改完四个维度后只写一句“模型变大所以更慢”。先固定三个，逐个改变一个并对照 shape 表。

## 从 CPU 实验迁移到 GPU/NPU

保留相同输入、权重、mask、RoPE 约定和 cache 语义，再换设备/算子后端。记录隐式 cast/copy、cache 格式与框架调度。设备是否支持某种矩阵指令或异步搬运不能从 CUDA 命名推断，需查实际 SDK。

完整 block 的数值对照是第一层；逐算子性能与 E2E 是第二层；生产框架的真实请求是第三层。这个实验覆盖前两层的可运行入口，不把它包装成 vLLM 或任意国产 NPU 的实测。

## 参考资料

- [Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：矩阵维度与计算/访存。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：设备分派和 mask 语义。
- [本仓性能测量章](../02-perf-aware-ops/benchmark-and-profiling.md)：如何将 profile 观察转成可证伪假设。
