# CI 与可复现交付

## 建议 CI 阶段

~~~text
format/lint
→ CPU reference tests
→ GPU correctness (可用 GPU runner 时)
→ sanitizer / race checks
→ microbenchmark (专用稳定 runner)
→ package/integration test
~~~

普通 CI runner 常没有 GPU，或 GPU 型号/频率不稳定。不要让普通 CI 的噪声 benchmark 阻塞合并；性能回归应使用固定专用设备并设合理容差。

## 版本记录

每次性能结果保存 Git commit、dirty 状态、GPU 名称和 compute capability、驱动/CUDA/框架版本、编译 flags、输入 shape/dtype/layout、warmup/迭代次数、统计方法、原始报告路径和结论。

## 本仓库已经提供的构建和测试入口

CUDA CMake 项目已为 vector_add、block_reduce_sum、naive_gemm、tiled_gemm、row_ops、operator_ladders 注册 CTest 项。配置、构建和运行是三步，各自可能失败：

~~~powershell
cmake -S examples/cuda -B build/cuda
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
~~~

配置失败常见于找不到 CUDA Toolkit 或宿主编译器；编译失败可能是语言/API/架构不匹配；运行失败则可能是设备不可用、驱动兼容或实际 kernel 错误。不能仅凭“CMake 配置成功”宣称样例可运行。

默认 CUDA architecture 使用 native，需在合适设备环境中配置；若交叉构建或构建机与部署 GPU 不同，通过 CMAKE_CUDA_ARCHITECTURES 明确指定与目标相容的架构。填写前查目标型号，不能复制别人的数值。

## 如何接入 CI

普通无 GPU runner 先执行标准 Python 测试，验证 CPU reference、在线合并与索引数据流；有 PyTorch 的 runner 再执行框架接入示例。GPU correctness/sanitizer 在受控 GPU runner 上运行，并记录设备版本；性能测试使用稳定专用 runner，以免共享环境波动掩盖回归。

新增练习应跑 `python exercises/check.py selftest` 和 `python exercises/reading/check.py --answers exercises/reading/solutions.json`，不要把未完成的学生答案直接接入仓库 CI。模型切片执行 `python -m unittest discover -s 06-model-slice -p "test_*.py" -v`；benchmark 与 profiler 留在独立的测量任务，不靠某个固定微秒数判定普通功能回归。

可把下表当作 CI job 设计，而不是把全部命令塞进一个无法判断失败层次的脚本：

| job | 前置条件 | 失败说明 | 产物 |
|---|---|---|---|
| cpu-semantics | Python | 算法/边界回归 | unittest 日志 |
| framework | PyTorch | 注册、fake、梯度或图模式问题 | 分项检查日志 |
| cuda-build | Toolkit + C++ 工具链 | 编译/API/架构问题 | 编译日志 |
| cuda-correctness | 兼容 GPU | 设备计算结果错误 | CTest 输出 |
| cuda-sanitizer | GPU + 检查工具 | 访问/竞争/同步问题 | sanitizer 报告 |
| performance | 稳定 GPU 与固定输入 | 相同条件下性能变化 | 原始样本和 profile |

当前仓库提供这些运行入口与课程说明，没有声称已配置远端 GPU runner。将模板接入实际 CI 时应把 runner 能力和凭据配置留在仓库/平台的正式设置中。

## 发布前如何做到别人能复现

README 应明确依赖和运行目录；性能记录关联唯一 commit；测试脚本使用可固定的 seed；输出中标注实际设备与计时范围。若性能记录来自 dirty 工作树，应保存 patch，否则别人拿同一 commit 也可能无法得到同一代码。

有缺失环境时写具体缺项，例如“本机 PyTorch CPU 测试通过，CUDA 编译/运行未执行”；不要写笼统的“测试通过”。新增后端应单独增加状态，不继承旧后端的验证结论。

## 练习：诊断三个交付故障

1. CI 没有 GPU，但 nvcc 能编译：cuda-build 可以执行，cuda-correctness 不应伪装成通过。
2. kernel correctness 通过，opcheck 失败：检查注册、别名、fake 与 autograd，不能只重复运行 kernel。
3. 同一 commit 的 benchmark 差 20%：先比设备/版本/频率/输入/计时范围和并发环境，再判断是不是代码回归。

这三题分别训练编译与运行边界、kernel 与框架边界、代码与测量环境边界。

## 参考资料

- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [PyTorch custom operator testing](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
