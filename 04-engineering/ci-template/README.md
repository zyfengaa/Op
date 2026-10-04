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

## 参考资料

- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [PyTorch custom operator testing](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
