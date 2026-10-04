# 10：完整项目实战：从需求到交付

## 项目目标

实现一个可集成的 fused GEMM：`Y = activation(A × B + bias)`，同时保留 CPU reference、naive CUDA、tiled CUDA 和框架调用入口。

## 项目阶段

1. 需求：定义 shape、dtype、layout、activation、误差和支持范围。
2. Reference：实现 CPU 版本，生成固定向量和随机测试。
3. Baseline：接入现有框架算子，记录 E2E 时间。
4. Kernel：实现 naive，再实现 tiled 版本。
5. 优化：逐步加入 coalescing、shared memory、register tile、MMA 或 fusion。
6. 验证：覆盖非整除 shape、不同 dtype、空/小输入和错误输入。
7. Profiling：保存每一版本的 metrics 和报告，写出观察—假设—验证。
8. 集成：加入 Python/C++ API、构建脚本、CI 测试和回归 benchmark。
9. 交付：README、限制说明、性能表、已知问题和复现命令。

## 最终验收标准

- 数学结果与 reference 在规定误差内一致。
- 所有支持的 shape 都有测试，不依赖“刚好整除 tile”。
- 优化前后 benchmark 可复现，不能只报告单次最好结果。
- 至少有一份 profiler 证据解释优化原因。
- 算子级加速能说明是否改善 E2E；若没有，要解释其他瓶颈。
- 公共语义与后端实现分离，迁移到另一后端时测试无需重写。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
