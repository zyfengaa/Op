# GEMM V2：Shared Memory Tiling 实验

教学实现见 [tiled_gemm.cu](tiled_gemm.cu)。它用于理解 block 合作加载、shared memory 复用和尾块处理；尚未在本机 CUDA 环境编译验证，因此这里不声称实测性能。

## 目标

将 V1 每个输出线程重复读取 A/B 的模式改为 block 协作装入 tile，再复用 tile 数据。

## 设计步骤

1. 选择 BM、BN、BK。
2. 推导每个 block 覆盖的输出区域和 CTA grid。
3. 计算 shared memory 需求：A tile 为 BM×BK，B tile 为 BK×BN。
4. 所有线程协作加载，越界 load 填零。
5. block barrier 确保加载完成。
6. 计算 accumulator tile。
7. barrier 确保计算不再读取旧 tile 后，进入下一轮。
8. 仅对有效 row/col 写回。

源码用固定 TILE=16，使用二维 block。线程 `(ty,tx)` 分别从 A 载入 `A[row, tile*K+tx]`，从 B 载入 `B[tile*K+ty, col]`。加载结束后第一次 barrier 保证整个 tile 可读；循环累加结束后第二次 barrier 保证没有线程仍在使用旧 tile。

## 推导练习

如果 BM=64、BN=128、BK=32、dtype=FP32：

- A tile bytes = BM×BK×4。
- B tile bytes = BK×BN×4。
- 估算一个 block 的共享内存，并核对设备上限。
- grid_m=ceil(M/BM)，grid_n=ceil(N/BN)。

## 验证矩阵

测试 M/N/K 各自分别为 tile-1、tile、tile+1；再测试大规整 shape。对比 V1 和 reference，保存 sanitizer 结果与 profiler 报告。

## 性能假设模板

观察 Global traffic 或重复 load；假设 tile reuse 会降低 HBM bytes；改变 tile 代码；验证 traffic、时间和 occupancy。如果 traffic 下降但时间没降，调查同步、shared bank conflict、launch 或资源限制。
