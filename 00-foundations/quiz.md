# 基础概念自测

先独立回答，再查参考资料。重点是解释理由，而不是背关键词。

1. 为什么极小数组的 GPU Vector Add 可能比 CPU 慢？
2. `N=1000, block=256` 时，grid 有多少 block？尾部有多少无效线程？
3. row-major `A[rows][cols]` 中 `A[r,c]` 的线性偏移是什么？
4. Warp 内线程地址连续为什么通常更有利？
5. Shared memory 与 register 的作用差别是什么？
6. 为什么 `__syncthreads()` 不能完成普通 grid 级同步？
7. Occupancy 低是否一定说明 kernel 慢？还需要看什么？
8. PyTorch opcheck 是否能证明梯度数学正确？为什么？
9. 什么时候应把 host-device copy 计入 benchmark？
10. 跨到另一种 GPU 时，哪些是算法概念，哪些是平台 API？

## 答题检查

- 能解释 launch、数据位置和工作量之间的关系。
- 能完整推导索引和地址，而不是凭印象猜。
- 能指出至少两项还需 profiler 或设备规格验证的事实。
- 能区分算子数学语义与后端实现细节。
