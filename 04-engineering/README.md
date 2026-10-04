# 04 · 工程化：从 kernel 到可维护算子

## 课程

- [接口和分层设计](interface-study/README.md)
- [测试框架](test-framework/README.md)
- [CI 与交付模板](ci-template/README.md)

## 工程目标

把算子语义、CPU reference、测试契约和 benchmark 与后端实现分开。CUDA 是一个 backend；Triton 或其他加速卡可以增加新 backend，而不复制所有测试和接口。

## 参考资料

- [PyTorch custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
