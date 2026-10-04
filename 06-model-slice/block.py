"""One inference-only decoder block with explicit prefill/decode semantics.

Reference: separate Q/K/V and gate/up projections plus materialized attention.
Optimized candidate: packed projections plus PyTorch SDPA (backend not forced).
No checkpoint, tokenizer, training, GQA, dropout, quantization or paged cache.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    width: int = 64
    heads: int = 4
    hidden: int = 128
    eps: float = 1e-6

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.hidden <= 0:
            raise ValueError("positive dimensions required")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with even head_dim")
        if self.eps <= 0:
            raise ValueError("positive epsilon required")


@dataclass(frozen=True)
class KVCache:
    k: torch.Tensor  # [B,H,L,Dh], already rotated
    v: torch.Tensor

    @property
    def length(self):
        return self.k.shape[2]


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def rope(x, positions):
    # Adjacent-pair convention; do not mix with a split-half convention checkpoint.
    d = x.shape[-1]
    inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angles = positions.to(x.dtype)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class TinyDecoderBlock(nn.Module):
    def __init__(self, config=Config(), seed=2026, device="cpu", dtype=torch.float32):
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(seed)
        def weight(rows, cols):
            data = torch.randn(rows, cols, generator=generator, dtype=torch.float64) / math.sqrt(rows)
            return nn.Parameter(data.to(device=device, dtype=dtype), requires_grad=False)
        self.w_qkv = weight(config.width, 3 * config.width)
        self.w_o = weight(config.width, config.width)
        self.w_gate_up = weight(config.width, 2 * config.hidden)
        self.w_down = weight(config.hidden, config.width)
        self.norm1 = nn.Parameter(torch.ones(config.width, device=device, dtype=dtype), requires_grad=False)
        self.norm2 = nn.Parameter(torch.ones(config.width, device=device, dtype=dtype), requires_grad=False)

    def forward(self, x, cache=None, variant="reference", profile=False):
        if variant not in ("reference", "optimized"):
            raise ValueError("unknown variant")
        cfg = self.config
        if x.ndim != 3 or x.shape[-1] != cfg.width or x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x must be nonempty [B,S,width]")
        if x.dtype != self.w_qkv.dtype or x.device != self.w_qkv.device:
            raise ValueError("input and weights must have the same dtype/device")
        batch, seq, _ = x.shape
        dh = cfg.width // cfg.heads
        past = 0
        if cache is not None:
            expected = (batch, cfg.heads, cache.length, dh)
            if tuple(cache.k.shape) != expected or tuple(cache.v.shape) != expected:
                raise ValueError("incompatible KV cache shape")
            if any(t.dtype != x.dtype or t.device != x.device for t in (cache.k, cache.v)):
                raise ValueError("cache dtype/device mismatch")
            past = cache.length
        def phase(name):
            return torch.profiler.record_function("slice/" + name) if profile else nullcontext()

        with phase("rmsnorm_attn"):
            normed = rms_norm(x, self.norm1, cfg.eps)
        with phase("qkv_projection"):
            if variant == "optimized":
                q, k, v = (normed @ self.w_qkv).chunk(3, dim=-1)
            else:
                wq, wk, wv = self.w_qkv.chunk(3, dim=-1)
                q, k, v = normed @ wq, normed @ wk, normed @ wv
            q, k, v = [t.reshape(batch, seq, cfg.heads, dh).transpose(1, 2) for t in (q, k, v)]
        with phase("rope"):
            positions = torch.arange(past, past + seq, device=x.device)
            q, k = rope(q, positions), rope(k, positions)
        with phase("kv_cache"):
            if cache is not None:
                k, v = torch.cat((cache.k, k), dim=2), torch.cat((cache.v, v), dim=2)
            new_cache = KVCache(k, v)
        with phase("attention"):
            # True = visible for SDPA boolean mask; absolute query positions matter.
            visible = torch.arange(past + seq, device=x.device)[None, :] <= positions[:, None]
            if variant == "optimized":
                z = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, dropout_p=0.0)
            else:
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
                probabilities = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)
                z = probabilities @ v
            z = z.transpose(1, 2).reshape(batch, seq, cfg.width)
        with phase("o_projection"):
            projected = z @ self.w_o
        with phase("residual_attn"):
            residual = x + projected
        with phase("rmsnorm_mlp"):
            normed = rms_norm(residual, self.norm2, cfg.eps)
        with phase("gate_up_projection"):
            if variant == "optimized":
                gate, up = (normed @ self.w_gate_up).chunk(2, dim=-1)
            else:
                wg, wu = self.w_gate_up.chunk(2, dim=-1)
                gate, up = normed @ wg, normed @ wu
        with phase("silu_multiply"):
            hidden = F.silu(gate) * up
        with phase("down_projection"):
            down = hidden @ self.w_down
        with phase("residual_mlp"):
            y = residual + down
        return y, new_cache


def shapes(config, batch, seq, past=0):
    m, d, h = batch * seq, config.width, config.hidden
    return {
        "qkv_gemm": {"M": m, "K": d, "N": 3*d},
        "o_gemm": {"M": m, "K": d, "N": d},
        "gate_up_gemm": {"M": m, "K": d, "N": 2*h},
        "down_gemm": {"M": m, "K": h, "N": d},
        "q": [batch, config.heads, seq, d // config.heads],
        "kv": [batch, config.heads, past + seq, d // config.heads],
        "logical_score_shape": [batch, config.heads, seq, past + seq],
    }
