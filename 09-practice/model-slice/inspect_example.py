"""Print a four-channel worked example; not a benchmark or pretrained model."""
# 逐通道工作台：用一组小手算输入把"输入 → RMSNorm → QKV → RoPE → causal mask →
# prefill/decode shape → 缓存对齐"这条链打印出来，供第一次读模型结构的人对照 walkthrough.md。
#
# 它是可复现的数值演示，不是 benchmark（没有计时），也不是预训练模型：权重随机、
# 无分词器、无词表 head。全部用 float64 跑，断言才敢收到 1e-10 这个量级。
import torch
from block import Config, TinyDecoderBlock, rms_norm, rope, shapes


@torch.inference_mode()
def main():
    # 单线程：小张量下让结果稳定可复现，避免线程调度带来的非确定性。
    torch.set_num_threads(1)
    # 固定打印精度、关掉科学计数法，方便逐行手算对照。
    torch.set_printoptions(precision=5, sci_mode=False)
    # 故意选极小维度（D=4, heads=2 → head_dim=2）：小到能整行打印、能手工复算。
    cfg = Config(width=4, heads=2, hidden=6)
    # float64：这是个"对照工具"，用双精度把浮点噪声压到可忽略，断言才敢用 1e-10。
    model = TinyDecoderBlock(cfg, dtype=torch.float64)
    # 4 个 token 的一段序列，第一维是 batch=1。
    x = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.],
                       [1., -1., 2., -2.], [2., 0., -1., 3.]]], dtype=torch.float64)
    # 前 3 个 token 当 prefill 前缀，最后 1 个当作要单独 decode 的 token。
    prefix, token = x[:, :3], x[:, 3:]
    normalized = rms_norm(prefix, model.norm1, cfg.eps)
    # packed QKV：一次矩阵乘后在最后一维切出 Q/K/V，与 block.py 的 optimized 路径一致。
    q, k, v = (normalized @ model.w_qkv).chunk(3, dim=-1)
    # [B,S,H,Dh]=[1,3,2,2] → [B,H,S,Dh]=[1,2,3,2]，这里只对 Q 拆头做演示。
    q = q.reshape(1, 3, 2, 2).transpose(1, 2)
    # 只用前 3 个绝对位置（0,1,2），这正是 prefill 的用法。
    rotated = rope(q, torch.arange(3))
    print("input prefix [B,S,D]:", tuple(prefix.shape))
    print("first row:", prefix[0, 0])
    print("first normalized row:", normalized[0, 0])
    print("Q, head 0 before RoPE:\n", q[0, 0])
    print("Q, head 0 after RoPE:\n", rotated[0, 0])
    # RoPE 是一次旋转：每个相邻元素对的平方和（即范数）必须不变，这是它保范数的直接证据。
    torch.testing.assert_close(q.square().sum(-1), rotated.square().sum(-1))
    # prefill 的可见性就是标准下三角：query i 只能看到 key 0..i。
    print("prefill visible mask:\n", torch.arange(3)[None, :] <= torch.arange(3)[:, None])
    # decode 只有一个 query，但它的绝对位置是 3，应该看见全部 4 个 key（0..3）。
    # 若误把它当成"局部下标 0 的下三角"，就只会看见第一个 key——这是最常见的 cache bug。
    print("decode visible mask:\n", torch.arange(4)[None, :] <= torch.tensor([[3]]))
    # 对照打印 prefill 与单 token decode 的 shape：M=B*S 从 3 降到 1，kv 长度从 3 涨到 4。
    print("prefill shapes:", shapes(cfg, 1, 3))
    print("decode shapes:", shapes(cfg, 1, 1, 3))
    # 参照系：整段一次前向（reference）。
    full, _ = model(x, variant="reference")
    # 只跑前缀，留下一个 K 已旋转好、长度为 3 的 cache。
    _, cache = model(prefix, variant="reference")
    # 用 optimized 路径对同一个 token 做增量 decode，看它是否等于"整段重算"的最后一位。
    decoded, updated = model(token, cache, variant="optimized")
    # 跨实现对拍（reference 整段 vs optimized 增量）是 cache 正确性最强的验收。
    torch.testing.assert_close(decoded, full[:, -1:], atol=1e-10, rtol=1e-10)
    print("last token, full reference:", full[0, -1])
    print("last token, cached candidate:", decoded[0, 0])
    print("max abs difference:", (decoded - full[:, -1:]).abs().max().item())
    print("old/new KV length:", cache.length, updated.length)
    # "逻辑" KV 字节数 = 元素数 × 每元素字节，是缓存内容的理论大小；
    # 它不等于进程 RSS，也不等于显存峰值（还有框架与分配器的开销）。口径别混着比。
    print("old/new logical KV bytes:",
          (cache.k.numel() + cache.v.numel()) * cache.k.element_size(),
          (updated.k.numel() + updated.v.numel()) * updated.k.element_size())
    print("worked example: PASS")


if __name__ == "__main__":
    main()
