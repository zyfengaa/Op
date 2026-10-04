# 05：Transpose：理解访存合并与 Shared Memory Bank Conflict

## 本章目标

通过转置这个计算量很小、访存特征明显的算子，学会区分“带宽高”和“有效带宽高”。

## 数学定义

```text
Y[col, row] = X[row, col]
```

naive 版本通常让线程连续读取 X，却跨行写 Y，或反过来，导致一侧访问不合并。

## Tiled Transpose

典型方案是把 tile 从 Global Memory 合并读取到 shared memory，再以另一种方向写出。shared tile 常使用 padding，例如 `[TILE][TILE+1]`，避免多个线程落到同一个 bank；实际是否需要 padding 取决于访问模式和硬件。

## 小案例

1. Naive transpose：记录读写两侧 transaction。
2. Tiled transpose：比较 coalesced load/store。
3. Padded tile：打开和关闭 padding，观察 bank conflict 指标。

## 验证方法

测试方阵、矩形矩阵、1×N、N×1 和非 tile 整除尺寸。检查 `Y[col,row] == X[row,col]`，再计算有效带宽 `2 × elements × bytes / time`。

## 常见错误

- 把输入和输出的 shape 仍当成相同方阵。
- 只修复 Global load，没有检查 Global store。
- 误以为 padding 永远更快；padding 也会增加 shared memory 占用。

## 参考资料

- [CUDA C++ Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [CUDA Programming Guide：Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
