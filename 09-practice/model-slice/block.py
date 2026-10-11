"""One inference-only decoder block with explicit prefill/decode semantics.

Reference: separate Q/K/V and gate/up projections plus materialized attention.
Optimized candidate: packed projections plus PyTorch SDPA (backend not forced).
No checkpoint, tokenizer, training, GQA, dropout, quantization or paged cache.
"""
# 这是一个"推理专用"的 decoder block 切片，用来把前面学到的算子串成一条数据流。
#
# 两种实现共享同一组权重，只改变算子的组合方式，方便做"等价性 + 性能"的对照：
#   * reference：Q/K/V 分开投影、gate/up 分开投影，attention 显式走
#     scores → softmax → P@V，每一步都看得见，适合逐行核对数值。
#   * optimized：把 QKV 合成一次矩阵乘、gate/up 合成一次矩阵乘，attention 直接调用
#     PyTorch SDPA。名字只是"候选"，并不保证每个尺寸都更快，后端也不能假定。
#
# 刻意不包含：checkpoint/分词器、训练、GQA、dropout、量化、分页 KV cache。
# 这些都会改变数值语义，引入后就不能再称其为"同一个 block 的优化"。
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    # frozen：配置是只读契约，避免被参数量推导或 profiler 顺手改写，
    # 同一份配置在整段对照里保持一致。
    width: int = 64
    heads: int = 4
    hidden: int = 128
    eps: float = 1e-6

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.hidden <= 0:
            raise ValueError("positive dimensions required")
        # width 必须能整除 heads（否则 head_dim 不是整数），
        # 且 head_dim 必须为偶数——RoPE 按相邻元素成对旋转，奇数通道无法配对。
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with even head_dim")
        # eps 进入 rsqrt 的分母，必须为正；否则全零输入会得到 inf/nan。
        if self.eps <= 0:
            raise ValueError("positive epsilon required")


# KV cache：decode 阶段复用的 K/V 历史，形状 [B,H,L,Dh]。
# k 保存的是已经做过 RoPE 的结果，所以追加新 token 时直接拼接即可；
# 既不能每次从 0 重新旋转，也不能把历史 K 再旋转一遍（那会变成双重旋转）。
# v 不做位置编码，保持投影后的原值。
@dataclass(frozen=True)
class KVCache:
    k: torch.Tensor  # [B,H,L,Dh], already rotated
    v: torch.Tensor

    @property
    def length(self):
        # 长度直接读张量形状，而不是单独维护计数器，避免两者不同步。
        return self.k.shape[2]


# RMSNorm：只按均方根缩放，不减均值。这与 LayerNorm 有本质区别——
# 对常量输入，LayerNorm 会输出 0，而 RMSNorm 输出缩放后的常数（见 test_block.py 的用例）。
# rsqrt(mean(x^2) + eps)：eps 防止全零向量导致除零。
def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


# RoPE：按"相邻元素成对"的约定做旋转，使用绝对位置。
# 两条关键契约：旋转保范数（每对元素的平方和不变，见 test_block.py），
# 位置必须取绝对位置——decode 时新 token 的 position 从已有长度 past 开始，
# 不能从 0 重来，否则同一 token 在不同上下文里会得到不同的编码。
# 这个相邻成对约定不能与 split-half 约定的 checkpoint 混用，混用会静默错位。
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
        # 固定 seed 生成权重：参考版与候选版共享同一组权重，数值/性能对照才有效。
        # 在 CPU 上用 float64 生成再转到目标 dtype，让权重在换设备/换精度时仍可复现。
        generator = torch.Generator(device="cpu").manual_seed(seed)
        def weight(rows, cols):
            # 除以 sqrt(rows) 把激活尺度稳定在 O(1)：否则多层堆叠后数值会随深度爆炸或消失。
            data = torch.randn(rows, cols, generator=generator, dtype=torch.float64) / math.sqrt(rows)
            return nn.Parameter(data.to(device=device, dtype=dtype), requires_grad=False)
        # packed 权重：QKV 拼成一块、gate/up 拼成一块。两种 variant 读的是同一块内存，
        # 所以 reference 的拆分计算不会因参数不同而让对照失真。
        self.w_qkv = weight(config.width, 3 * config.width)
        self.w_o = weight(config.width, config.width)
        self.w_gate_up = weight(config.width, 2 * config.hidden)
        self.w_down = weight(config.hidden, config.width)
        # requires_grad=False：这是推理切片，不训练，不需要梯度。
        self.norm1 = nn.Parameter(torch.ones(config.width, device=device, dtype=dtype), requires_grad=False)
        self.norm2 = nn.Parameter(torch.ones(config.width, device=device, dtype=dtype), requires_grad=False)

    def forward(self, x, cache=None, variant="reference", profile=False):
        # variant 只允许这两个取值；名字不承诺性能，只描述算子组合方式。
        if variant not in ("reference", "optimized"):
            raise ValueError("unknown variant")
        cfg = self.config
        # 输入契约：非空 [B,S,width]。S=0 的空序列会让 attention mask 退化，
        # 这里直接拒绝，而不是让它产生 NaN 再往下传。
        if x.ndim != 3 or x.shape[-1] != cfg.width or x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x must be nonempty [B,S,width]")
        # 权重与输入必须同 dtype/同设备，否则矩阵乘会静默隐式转换或直接报错。
        if x.dtype != self.w_qkv.dtype or x.device != self.w_qkv.device:
            raise ValueError("input and weights must have the same dtype/device")
        batch, seq, _ = x.shape
        dh = cfg.width // cfg.heads
        past = 0
        if cache is not None:
            # 校验 cache 的形状/dtype/device 与本次调用匹配，避免拼出错误长度或静默广播。
            expected = (batch, cfg.heads, cache.length, dh)
            if tuple(cache.k.shape) != expected or tuple(cache.v.shape) != expected:
                raise ValueError("incompatible KV cache shape")
            if any(t.dtype != x.dtype or t.device != x.device for t in (cache.k, cache.v)):
                raise ValueError("cache dtype/device mismatch")
            past = cache.length
        def phase(name):
            # 打点与否由 profile 决定：普通计时跑未插桩的完整路径，profile 另跑一次，
            # 两者不混合，避免把插桩开销算进性能数字。
            return torch.profiler.record_function("slice/" + name) if profile else nullcontext()

        with phase("rmsnorm_attn"):
            normed = rms_norm(x, self.norm1, cfg.eps)
        with phase("qkv_projection"):
            if variant == "optimized":
                # 合成一次矩阵乘（[M,D] @ [D,3D]），减少 kernel 数量与内存往返。
                q, k, v = (normed @ self.w_qkv).chunk(3, dim=-1)
            else:
                # 分开三次投影，读的是同一块 packed 权重，便于逐项核对。
                wq, wk, wv = self.w_qkv.chunk(3, dim=-1)
                q, k, v = normed @ wq, normed @ wk, normed @ wv
            # [B,S,H,Dh] → [B,H,S,Dh]：把序列维挪到头维度之后，attention 在最后两维做。
            q, k, v = [t.reshape(batch, seq, cfg.heads, dh).transpose(1, 2) for t in (q, k, v)]
        with phase("rope"):
            # 绝对位置从 past 开始：prefill 时 past=0，decode 时 past=已有缓存长度。
            positions = torch.arange(past, past + seq, device=x.device)
            q, k = rope(q, positions), rope(k, positions)
        with phase("kv_cache"):
            if cache is not None:
                # 沿长度维拼接历史 K/V。torch.cat 产生新张量、不原地改旧 cache，
                # 因此基准 cache 可在多次计时迭代中反复复用（代价是多一次拷贝）。
                k, v = torch.cat((cache.k, k), dim=2), torch.cat((cache.v, v), dim=2)
            new_cache = KVCache(k, v)
        with phase("attention"):
            # True = visible for SDPA boolean mask; absolute query positions matter.
            # 用绝对位置比较：query 的绝对下标是 past+i，key 的下标是 0..past+seq-1。
            # 不能对矩形 attention 简单套左上三角——decode 时 query 的局部下标是 0，
            # 但绝对位置是 past，误当三角会只让最后一个 token 看见第一个 key。
            visible = torch.arange(past + seq, device=x.device)[None, :] <= positions[:, None]
            if variant == "optimized":
                # 交给 SDPA，由 PyTorch 按环境分派后端；调了它不等于跑了 FlashAttention。
                z = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, dropout_p=0.0)
            else:
                # 显式路径：缩放点积 → 掩码（不可见处置 -inf）→ softmax → 乘 V。
                # 先掩码再 softmax，保证被屏蔽位置的权重严格为 0，而不是"很小"。
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
                probabilities = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)
                z = probabilities @ v
            # [B,H,S,Dh] → [B,S,H,Dh] → [B,S,D]：换回原布局好做输出投影。
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
            # SwiGLU：SiLU(gate) 逐元素乘 up，是 MLP 里的非线性来源。
            hidden = F.silu(gate) * up
        with phase("down_projection"):
            down = hidden @ self.w_down
        with phase("residual_mlp"):
            y = residual + down
        return y, new_cache


# 把这次调用的 shape 显式写出来，供报告引用。M=B*S 是各 GEMM 的行数：
# prefill 时 M=B*S（每次整段），decode 单 token 时 M=B（batch=1 才是 1）。
# kv 的长度是 past+seq；logical_score_shape 是"不做融合时会生成的完整 scores 矩阵"形状，
# 用来解释 attention 为什么是显存与带宽的关键点。
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
