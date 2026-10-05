"""Print a four-channel worked example; not a benchmark or pretrained model."""
import torch
from block import Config, TinyDecoderBlock, rms_norm, rope, shapes


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, sci_mode=False)
    cfg = Config(width=4, heads=2, hidden=6)
    model = TinyDecoderBlock(cfg, dtype=torch.float64)
    x = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.],
                       [1., -1., 2., -2.], [2., 0., -1., 3.]]], dtype=torch.float64)
    prefix, token = x[:, :3], x[:, 3:]
    normalized = rms_norm(prefix, model.norm1, cfg.eps)
    q, k, v = (normalized @ model.w_qkv).chunk(3, dim=-1)
    q = q.reshape(1, 3, 2, 2).transpose(1, 2)
    rotated = rope(q, torch.arange(3))
    print("input prefix [B,S,D]:", tuple(prefix.shape))
    print("first row:", prefix[0, 0])
    print("first normalized row:", normalized[0, 0])
    print("Q, head 0 before RoPE:\n", q[0, 0])
    print("Q, head 0 after RoPE:\n", rotated[0, 0])
    torch.testing.assert_close(q.square().sum(-1), rotated.square().sum(-1))
    print("prefill visible mask:\n", torch.arange(3)[None, :] <= torch.arange(3)[:, None])
    print("decode visible mask:\n", torch.arange(4)[None, :] <= torch.tensor([[3]]))
    print("prefill shapes:", shapes(cfg, 1, 3))
    print("decode shapes:", shapes(cfg, 1, 1, 3))
    full, _ = model(x, variant="reference")
    _, cache = model(prefix, variant="reference")
    decoded, updated = model(token, cache, variant="optimized")
    torch.testing.assert_close(decoded, full[:, -1:], atol=1e-10, rtol=1e-10)
    print("last token, full reference:", full[0, -1])
    print("last token, cached candidate:", decoded[0, 0])
    print("max abs difference:", (decoded - full[:, -1:]).abs().max().item())
    print("old/new KV length:", cache.length, updated.length)
    print("old/new logical KV bytes:",
          (cache.k.numel() + cache.v.numel()) * cache.k.element_size(),
          (updated.k.numel() + updated.v.numel()) * updated.k.element_size())
    print("worked example: PASS")


if __name__ == "__main__":
    main()
