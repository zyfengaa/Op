# 论文与源码导航

从哪读起。每条给出**读它的理由**，不是罗列。

## 论文

| 论文 | 读它的理由 | 前置 |
|---|---|---|
| [Roofline](https://dl.acm.org/doi/10.1145/1498765.1498785) | 算术强度与性能上界的原始定义。读它能自己算「还有多少空间」 | [analysis](../04-performance/analysis.md) |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 不物化完整分数矩阵这个想法本身，比实现细节更重要 | [attention](../03-operators/attention/README.md) |
| [Online normalizer / softmax](https://arxiv.org/abs/1805.02867) | 单遍扫描算 softmax 的 m/l 重缩放。FlashAttention 的前身思想 | [softmax](../03-operators/normalization/softmax/walkthrough.md) |
| [Welford 的在线方差](https://dl.acm.org/doi/10.1145/359146.359153) | LayerNorm/RMSNorm 多块统计合并的公式来源 | [layernorm](../03-operators/normalization/layernorm/walkthrough.md) |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | 旋转位置编码。注意相邻成对 vs split-half 两种约定不可混用 | [model-slice](../09-practice/model-slice/walkthrough.md) |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 逐层误差补偿的量化方法 | [quantization](../07-quantization/algorithms.md) |
| [AWQ](https://arxiv.org/abs/2306.00978) | 按激活重要性保护权重通道 | [quantization](../07-quantization/algorithms.md) |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 把激活的量化难度迁移到权重 | [quantization](../07-quantization/algorithms.md) |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 分页 KV cache。理解推理系统里的内存管理 | [matmul/decode-gemm](../03-operators/matmul/decode-gemm.md) |

**读论文的纪律**：先读摘要和方法，判断它解决的是什么约束下的问题，再看实现。很多论文的收益依赖特定硬件代际或特定 shape。

## 开源 kernel 源码

| 项目 | 读它的理由 | 建议入口 |
|---|---|---|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | 生产级 GEMM 的事实标准。它的分层（threadblock / warp / thread）是 Tile 思想的完整实现 | 先读 docs 里的 GEMM 教程，再看 `include/cutlass/gemm/` |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | online softmax 的完整实现，含 forward 与 backward | `csrc/flash_attn/` |
| [Triton](https://github.com/triton-lang/triton) | 更高级的编程模型，用 tile 而不是线程思考 | `python/tutorials/` |
| [PyTorch ATen](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) | 看框架自己的 kernel 怎么写、怎么分派 | `cuda/` 目录下挑一个简单的 elementwise 起步 |
| [oneDNN](https://github.com/oneapi-src/oneDNN) | CPU 侧的高性能算子库 | 先看它的 benchdnn |

## 阅读方法

1. **先看接口再看实现。** 函数签名告诉你它承诺了什么语义。
2. **带着一个具体问题读。** 「它的 tile 是多大、为什么」比「通读一遍」有效。
3. **对照本仓的对应算子。** 本仓的教学实现刻意写得简单，读完再看生产实现，能看清哪些复杂度是为了性能。
4. **记下成立条件。** 每个实现都建立在假设之上（对齐、dtype、shape 范围）。找到那些假设，才知道什么时候不能照搬。

## 官方文档

见 [external-links.md](external-links.md)。

## 相关页面

- [decision-tree](decision-tree.md)：先判断要不要自己写
- [common-pitfalls](common-pitfalls.md)：读源码时对照常见错误
- [03-operators](../03-operators/README.md)：本仓的算子实现
