# Vector Add 的一次完整开发经历

任务：为连续向量相加提供正确实现，解释不同并行分配的代价，修好尾部/stride 错误，拿出真实测量，再把契约带到第二后端。先读 [公式与首次实现](walkthrough.md)，然后按本页依次动手。

## ① 建立能复现的 naive 基线

[operator_ladders.cu](../../examples/cuda/operator_ladders.cu) 的 add_v0 用一个 GPU 线程串行扫描所有元素。它刻意暴露“代码搬到 GPU 不等于获得并行性”，同时提供易懂的基线。host 生成随索引变化的输入，避免常量输入掩盖读错位置；每个版本都与相同 CPU 公式对照。

支持范围：本例为正长度的连续 FP32、独立输出；N=0 在 host 返回空结果，不启动零大小 grid。程序的内置数据是可准确表示的四分之一/二分之一值，不能因此省略其他数据分布的回归。

## ② 实际可执行的版本阶梯

| 版本 | 如何分配工作 | 希望解决的问题 | 代价或不保证的地方 |
|---|---|---|---|
| add_v0 | 单线程循环 | 建立基线 | 几乎没有设备并行度 |
| add_v1 | 一线程一元素、ceil grid | 暴露大量独立工作 | 大输入产生更多 block |
| add_v2 | grid-stride，示例封顶 64 个 block | 控制发射规模、每线程多元素 | 上限依设备/shape；未必优于 V1 |

代码确实提供三种 kernel，而不是三个设计标题。64 是教学参数，不是建议所有设备采用的最优值。V2 没有引入数据复用，只改变调度粒度；不要把它解释为更少算法字节数。

### 用 10 个元素看清线程分工

为手算把 blockDim 暂设为 4（可运行程序仍使用 256）。V1 启动 3 个 block：全局线程 0–9 各写一项，10–11 被 mask 掉。V2 假设只发射 2 个 block，grid stride=2×4=8：线程 0 写 0、8；线程 1 写 1、9；线程 2–7 各写一项。总元素仍恰好覆盖一次。

“每线程多算一点”不等于少读了输入。两个版本都需要读 10 个 A、10 个 B，写 10 个 C。V2 可能改变 block 调度和每线程循环成本，但不会把 120 字节的 FP32 算法流量变成 60 字节。这也是分析优化时必须区分工作分配与数据复用的原因。

真正扫参数时先固定 N，只改变 block/grid，再固定配置改变 N；保留比旧版更慢的记录。不能用小输入测 V1、大输入测 V2 后拿微秒数直接比较。

~~~powershell
cmake -S examples/cuda -B build/cuda
cmake --build build/cuda --config Release
.\build\cuda\Release\operator_ladders.exe
.\build\cuda\Release\operator_ladders.exe --benchmark
~~~

默认覆盖 0、1、255、256、257、4097、65539。benchmark 比较最大尺寸，排除分配/复制，计入每个版本的全部 kernel。当前环境没有 nvcc，CUDA 版本未在本机编译或运行，不能从代码推断谁胜出。

## ③ 四次“找 bug”任务

~~~powershell
python exercises/check.py show VA01
python exercises/check.py grade VA01
python exercises/check.py grade VA02
python exercises/check.py grade VA03
python exercises/check.py grade VA04
~~~

分别定位网格漏尾部、stride 遗失、广播轴错误和向量加载尾部缺失。修复写在 exercises/fault_injection/student.py。先让全部回归输入通过，再解释为何你的修改不会产生重复写或越界读。

## ④ 用真实报告建立预期

无需 GPU，可先执行：

~~~powershell
python labs/operator_story.py --output labs/results/local
~~~

仓库保存了 [实际 CPU 原始样本](../../labs/results/cpu-reference/summary.json)、[原生 profiler 输出](../../labs/results/cpu-reference/native-cpu.profiler.txt)、[trace](../../labs/results/cpu-reference/native-cpu.trace.json) 和 [逐项解读](../../labs/results/cpu-reference/README.md)。这些是 CPU 证据，不是 ncu 报告。

预期应先落在可检验的结构上：原生路径有 aten::add；65536 个 float64 输出逻辑大小为 524288 字节。profiler 的有仪器时间与无 profiler 基准时间不应直接相除。CPU 记录里小 N 的 Python 循环反而更快，大 N 才由原生库明显领先，见 [性能反例档案](../../labs/performance-counterexamples.md)。

在 CUDA 机器上对 add_v0/v1/v2 分别采集 ncu，先看 launch 配置、访问模式与 duration；期望 V0 并行度不足，V1/V2 连续访问。这里给的是待验证假设，没有填造“正常带宽 80%”之类的百分比。具体命令与解释见 [测量章节](../../02-perf-aware-ops/benchmark-and-profiling.md)。

## ⑤ 做一次真实的第二后端比较

[operator_story.py](../../labs/operator_story.py) 使用同一组有限 double 值比较 Python reference 与 PyTorch 原生 CPU add，并逐元素核验。转换输入/输出的版本单独计时，避免把格式转换成本隐藏起来。它是可运行的第二实现后端，不声称已验证某种 SIMD 指令。

迁移时保留输入输出和错误契约，重新决定布局、线程组、尾部、编译与同步。将来接入 CUDA/Triton 时，还需将 dtype 明确匹配，不能把本页 float64 CPU 测量与 float32 CUDA 计时当成等价比赛。

## 与模型主线的连接

残差 X+AttentionOutput 就是同形逐元素加法；bias 广播属于更复杂的逐元素地址映射。进入 [model slice](../../06-model-slice/README.md)，在 profiler 中找到 residual_attn/residual_mlp，再比较独立加法微基准和它们在整块里的占比。

## 交付判定

交付三版本正确性记录、VA01–VA04 的修复与解释、一份你自己跑的 CPU 或 GPU 原始报告、报告中的三个可核查结论，以及迁移契约。没有设备时标明 CUDA 待验证；读过源码不能代替运行证据。

## 参考资料

- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：线程层级与索引。
- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：基准范围、重复采样与线程设置。
