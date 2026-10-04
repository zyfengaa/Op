# Op：面向性能分析的算子开发

这个仓库不以“手写很多 Kernel”为目标，而是训练一条可以复用的算子性能工程闭环：

```text
Profiling
  → 定位 Top Kernel
  → 映射模型结构
  → 分析 M/N/K、dtype、layout
  → 计算 FLOPs、访存量和算术强度
  → Roofline 定性
  → 建立指标到根因的证据链
  → 提出优化假设
  → 实现 / 让 AI 辅助实现
  → 正确性验证
  → Micro Benchmark
  → Profiler 复核
  → E2E Benchmark
```

## 当前内容

- [从零开始的学习路线](docs/00_learning_path.md)：面向没有算子开发经验的开发人员，解释每个阶段要学什么、做什么实验、交付什么结果。
- [GEMM 优化阶梯实验](docs/02_gemm_optimization_ladder.md)：从 naive GEMM 逐步走到 coalescing、tiling、shared memory、register tiling、Tensor Core 和 pipeline。
- [Decode 小 M GEMM 性能分析](docs/01_decode_gemm_performance_analysis.md)：完整模拟一次 vLLM Decode 场景，解释为什么不能看到 Tensor Core 利用率低就直接优化 Tensor Core。
- [模拟指标计算脚本](examples/simulate_gemm_analysis.py)：只依赖 Python 标准库，计算 FLOPs、权重大小、算术强度、Roofline 分界点和阶段收益。
- [最小 GEMM 学习程序](examples/gemm_learning_demo.py)：只依赖 Python 标准库，演示 naive 与 tiled GEMM 的数学等价性、边界处理和结果校验。

运行示例：

```powershell
python examples/simulate_gemm_analysis.py
python examples/gemm_learning_demo.py
```

## 重要边界

文档中的 `320 us`、`82 us`、`1.5 TB/s`、`72%` 等数据是教学用模拟输入，不代表某个具体 GPU、模型或真实测量结果。迁移到真实项目时，必须记录：模型版本、batch/token shape、dtype、量化方式、硬件型号、驱动/编译器、Kernel 配置、采样次数和端到端测量方法。

## 学习原则

每个实验都必须完成四件事：先写清数学定义和输入输出，再实现一个最小版本；用 reference 和误差标准验证正确性；最后记录性能、硬件指标和“指标 → 根因 → 修改”的证据链。

CUDA 是训练场，不是能力边界。学会的应当是算子语义、并行算法、数据布局、硬件映射、性能分析和工程验证；CUDA、Triton、AscendC 或其他后端只是不同的实现层。
