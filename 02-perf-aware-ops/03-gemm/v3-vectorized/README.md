# GEMM V3：向量化 Load/Store

## 学习目标

理解更宽的每线程 load 可能减少指令数或更有效利用内存 transaction，但它依赖地址对齐、连续访问、尾部 mask 和编译器生成代码。

## 前置条件

- V2 的访存和正确性已理解。
- 指针与每行 stride 的对齐情况已知。
- 当前设备支持目标向量宽度。
- 尾部元素有 scalar/masked fallback。

## 实验方法

对比标量 load 与向量化 load，固定 shape、tile、编译 flags 和频率条件。检查生成代码/指令、transaction、带宽和绝对 kernel latency。若只有规整 shape 更快，应采用 shape-specialized dispatch 或明确限制支持范围。

## 常见失败

- 假设 base pointer 对齐但实际切片偏移。
- 跨越行尾读取下一行数据。
- 向量化后寄存器压力增加，occupancy 下降。
- 数据量太小，测量噪声大于差异。
