# 性能测量实战：从“快了多少”到“为什么快”

这章紧接 Softmax、LayerNorm 和 GEMM。前置知识是知道 kernel 的输入/输出、线程映射以及读取了哪些数据。实验使用 [row_ops.cu](../assets/cuda/row_ops.cu)，包含 CPU correctness reference、CUDA event 计时和便于 profiler 采集的运行模式。

## 1. 先定义你真正想缩短的时间

同一个程序可以有四个不同的时间：

| 时间范围 | 包含什么 | 回答的问题 |
|---|---|---|
| 宿主 launch 调用 | 提交工作的 CPU 时间 | 主机是否来不及提交工作 |
| 单个/一组 GPU kernel | 设备上的执行及范围内间隔 | kernel 实现是否改善 |
| 算子调用端到端 | 参数检查、分配、调度、必要同步等 | 应用调用一次的成本 |
| 模型/业务请求 | 前后处理、多个算子及通信 | 用户是否真正获益 |

如果输入原本就在 GPU，比较 kernel 时不应为其中一版额外加入 H2D；如果真实业务每次确实要从 CPU 传数据，最终验收就应包含这笔成本。正确的计时范围来自使用场景，不存在一律排除或一律包含传输的规则。

本仓库 --benchmark 打印的是设备事件范围内 100 次 launch 的批平均时间，不包括分配和复制。它提供一个教学基线；若要形成稳定性能结论，应对整个测量批次重复采样，并保存各批次原始结果。

## 2. 为什么普通秒表经常测到提交时间

下面这种思路容易误导：

~~~text
start = CPU 时间
launch()
end = CPU 时间
~~~

当 launch 异步返回时，end 时 GPU 可能还没有结束。若下一行开始复制结果，该复制/同步调用承担了等待，看上去就成了“kernel 很快，复制很慢”。

若使用宿主秒表，应在起点前排空需要排除的旧任务，并在终点前等待目标任务完成。若使用 CUDA events，则将 start/stop 事件放在需要测量的同一 stream 上，等待 stop 完成，再读取二者间的设备时间。跨 stream 的依赖与事件位置还需额外设计。

## 3. 读懂示例中的计时代码

row_ops 在 correctness 通过后做 20 次预热，等待预热结束，再执行：

~~~cpp
cudaEventRecord(start);
for (int i = 0; i < 100; ++i) launch();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
~~~

ms/100 是每次 launch 的批平均设备时间。循环里也有宿主提交动作，如果 GPU 很快而 CPU 提交不及时，设备时间范围内可能出现空隙，因此它不是保证不受宿主影响的纯指令吞吐测量。要知道有没有空隙，需要看时间线。

预热用于减少首次初始化、代码加载和初始 cache 状态带来的影响，但重复同一组输入也会形成较热 cache。若业务输入工作集远大于 cache，应该设计相应的轮换输入实验；不要把热数据微基准直接当成冷数据吞吐。

## 4. 先测能跑通的版本，再检查内存与同步

在仓库根目录构建后：

~~~powershell
ctest --test-dir build/cuda -C Release --output-on-failure
compute-sanitizer --tool memcheck .\build\cuda\Release\row_ops.exe
compute-sanitizer --tool racecheck .\build\cuda\Release\row_ops.exe
compute-sanitizer --tool synccheck .\build\cuda\Release\row_ops.exe
~~~

单配置生成器的可执行文件通常直接在 build/cuda 下。memcheck、racecheck、synccheck 分别帮助排查内存访问、共享存储竞争和同步使用问题；具体工具覆盖范围见官方文档。工具没有报告问题也不能证明所有输入都正确，仍需数值 reference 与有针对性的边界测试。

本例一次默认运行包含 12 个宽度，每个宽度各测 Softmax 和 LayerNorm；同一个宽度有普通值、常量值、较大偏移值三种行。新增 dtype/layout 时，应同步扩展测试矩阵。

## 5. 统计量：别只报告最小值

假设七次独立批次的平均耗时为 [8.0,8.1,8.1,8.2,8.3,8.4,14.0] 微秒。这是教学假设数据。最小值 8.0 体现接近理想的一个样本；中位数 8.2 对单个高值更稳健；14.0 可能来自调度、时钟或其他任务干扰，应该查原因并保留记录。

只删掉慢样本而不声明规则，会把正常波动隐藏起来。比较两个版本时应尽量在相同环境下交替运行，记录原始样本、统计方法和波动范围。当改善幅度和波动同一量级时，结论应是证据不足，而不是稳定提速。

现有 benchmark 尚未自动输出这些批次分布。练习可在它之外增加重复批次与 CSV 记录，明确区分批平均值和单次 latency 的分布。

## 6. 从工作量计算有效性能

Vector Add 的 FLOPs 约 N，FP32 读写约 12N 字节，算术强度约 1/12 FLOPs/byte。GEMM 的 FLOPs 约 2MNK，但流量依赖 tile/cache/布局，不能只看输入矩阵总大小。

定义：
- 有效 GFLOPS = 算法 FLOPs / 时间秒数 / 10^9。
- 有效 GB/s = 算法字节数 / 时间秒数 / 10^9。
- 硬件实测流量 = profiler 的特定内存层级计数，需注明层级。

FP32 行 Softmax 的理想读入+写出字节数约 8BD。若源码重读输入三次，源码级字节请求模型约 16BD；这两个分母得到不同的“带宽”，都不应冒称实际 HBM 计数器值。

## 7. Roofline 推理怎样使用才有意义

简化模型为 P <= min(P_peak, BW_peak * I)，其中 P 是 FLOPs/s、BW 是 byte/s、I 是 FLOPs/byte。假设某设备的相应计算路径上限是 100 TFLOPs/s，带宽上限为 1 TB/s，当 I=2 时，带宽线给出 2 TFLOPs/s；当 I=200 时，计算线给出 100 TFLOPs/s。

这些数字是演算，不代表任何具体设备。真实设备的 FP32 标量、FP16 矩阵指令、稀疏计算等有不同峰值，不能拿某种 dtype 的宣传峰值去评判另一种路径。

Roofline 给的是上界与方向，不解释所有损失。小 kernel 还可能受 launch、内存延迟、可用并行度、依赖链、指令发射或同步限制。看到带宽只达 30% 不能直接断定“带宽不是瓶颈”，也不能仅凭 occupancy 低就认定它是根因。

## 8. 采集一个明确的 kernel

先查看本机工具版本、可用 section 和 metrics，再使用示例模式：

~~~powershell
ncu --version
ncu --list-sections
ncu --kernel-name regex:row_softmax --launch-count 1 --set full -o softmax-profile .\build\cuda\Release\row_ops.exe --profile
~~~

--profile 只执行一个具有代表性的 128×1024 Softmax 和 LayerNorm，避免 profiler 对全部 correctness 形状和预热循环反复采集。过滤名称后仅采集 Softmax，报告保存为工具对应格式。full 采集可能涉及多次 replay，因此这次运行用于诊断，常规计时应另外不带 profiler 执行。

工具的指标和 section 随版本、设备变化。遇到 unsupported metric，先查询本机可用项，而不是照抄别的架构报告里的指标名。

## 9. 从证据到假设：三组阅读练习

情形 A：大量很小 kernel，每个计算量很少，时间线上有明显空隙。假设是提交/launch 成本占重要部分。可尝试融合或批处理，验证 launch 数与完整调用时间是否下降。不能仅因为 kernel 小就直接改算法。

情形 B：每线程访问跨度很大，算法有效字节远小于计数器报告的传输字节。假设是访存事务利用率差。先改线程到地址的映射，再比较事务和时间；若数据已在 cache，显存层面的变化可能不明显，应看对应层级。

情形 C：增大 tile 后理论输入复用更好，但实际变慢，同时寄存器/local memory 访问上升。假设是资源压力或 spill 抵消复用收益。尝试减小寄存器片或调整 block，并重新测试。不能只把 occupancy 的变化作为唯一解释。

每种情形都应保存“观察、假设、只改了什么、出现了什么结果”。假设被证伪也有价值，能避免下次重复试同样的无效优化。

## 10. 端到端收益的上限

假设一个待优化算子占完整请求时间的 20%，把它加速到 2 倍，原总时间 1 变成 0.8+0.2/2=0.9，整体约加速 1.11 倍。即使这个算子成本变为 0，总体也只能到 1/0.8=1.25 倍。这里还忽略了可能新增的转换/调度开销。

因此优化的优先级同时取决于算子占比、可实现收益和工程成本。profile 占比来自哪个输入、哪个阶段、哪个并发条件必须清楚；一次 profile 的热点不能保证覆盖全部工作负载。

## 11. 本章作业与验收

使用同一版 row_ops，在有 CUDA 的设备上提交 correctness 日志、sanitizer 输出、一份不带 profiler 的时间记录、一份 Softmax profiler 报告。给出一个具体优化假设并说明如果它错了，预计会观察到什么。

没有 GPU 时，完成字节数、算术强度和端到端上限的手算，将设备结果保持为“待实测”。不要拿 CPU 数据流模拟的运行时间填 GPU 性能表。

## 参考资料与阅读任务

- [CUDA Best Practices：Timing 与 Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)：核对计时范围和有效带宽定义。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：查过滤、采集、报告与本机 metrics 查询。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：明确每种检查的覆盖范围。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：观察 CPU/GPU 时间线与 launch 间隔。
