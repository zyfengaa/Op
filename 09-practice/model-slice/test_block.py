# test_block.py：对 TinyDecoderBlock 的可判定验收，共 7 项，与 README 的清单一一对应。
# 全部用 float64、小尺寸、固定 seed，容差收到 1e-10，让任何数值或语义错误都藏不住：
#   1. packed/SDPA 路径与显式路径数值一致；
#   2. 先前缀再后缀 == 完整 causal 前向的后缀，且旧 cache 不被改写；
#   3. 逐 token 增量 decode 的拼接 == 完整输出；
#   4. 改变未来 token 不影响过去输出（causal 属性）；
#   5. RoPE 保范数且使用绝对位置；
#   6. 常量输入的 RMSNorm 不等同于减均值的 LayerNorm；
#   7. 非法配置、空输入、错误 cache 被拒绝。
# 只覆盖推理切片，不验证训练梯度。
import unittest
import torch
from block import Config, KVCache, TinyDecoderBlock, rms_norm, rope


class BlockTest(unittest.TestCase):
    def setUp(self):
        # 单线程 + 固定 seed：测试必须确定性，失败要能稳定复现。
        torch.set_num_threads(1)
        # 极小尺寸（D=16, heads=2 → head_dim=8）配 float64：只为测准语义，不代表真实规模。
        self.model = TinyDecoderBlock(Config(16, 2, 24), dtype=torch.float64)
        g = torch.Generator().manual_seed(11)
        self.x = torch.randn(2, 7, 16, generator=g, dtype=torch.float64)

    def test_packed_sdpa_matches_explicit(self):
        # 同一组权重、同一输入，两种算子组合必须逐元素一致——这是把 optimized 当成
        # "等价候选"的前提；这里过不了，后面的性能数字就无意义。
        with torch.inference_mode():
            a, _ = self.model(self.x, variant="reference")
            b, _ = self.model(self.x, variant="optimized")
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_prefill_then_decode_matches_full_and_does_not_mutate_cache(self):
        # 先跑前缀再跑后缀，结果必须等于一次性整段前向的后缀。
        # 同时断言存下的旧 K 未被改写：cache 不可变，才能被多次计时迭代安全复用。
        with torch.inference_mode():
            for variant in ("reference", "optimized"):
                full, _ = self.model(self.x, variant=variant)
                _, cache = self.model(self.x[:, :4], variant=variant)
                saved = cache.k.clone()
                suffix, extended = self.model(self.x[:, 4:], cache, variant)
                torch.testing.assert_close(suffix, full[:, 4:], atol=1e-10, rtol=1e-10)
                torch.testing.assert_close(cache.k, saved, atol=0, rtol=0)
                # 长度契约：旧缓存停在 4，新缓存延伸到 7。
                self.assertEqual(cache.length, 4)
                self.assertEqual(extended.length, 7)

    def test_every_token_incremental_matches_full(self):
        # 一次只喂一个 token，逐次拼接的结果与整段一致。
        # 这是对"绝对位置 + cache 拼接"最直接的遍历：位置一旦重置或重复旋转，这里立刻崩。
        with torch.inference_mode():
            full, _ = self.model(self.x, variant="optimized")
            cache, pieces = None, []
            for token in range(self.x.shape[1]):
                output, cache = self.model(self.x[:, token:token + 1], cache, "optimized")
                pieces.append(output)
            torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-10, rtol=1e-10)

    def test_future_tokens_do_not_change_past_outputs(self):
        # causal 属性：改动第 4 个 token 之后的内容，前 4 个输出必须一动不动。
        # 这条也能抓出 mask 用错（比如看得到未来）导致的因果泄漏。
        changed = self.x.clone()
        changed[:, 4:] += 100
        with torch.inference_mode():
            for variant in ("reference", "optimized"):
                first, _ = self.model(self.x, variant=variant)
                second, _ = self.model(changed, variant=variant)
                torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-10, rtol=1e-10)

    def test_rope_preserves_pair_norm_and_uses_absolute_position(self):
        # 保范数：旋转前后每对元素的平方和相等。
        x = self.x.reshape(2, 7, 2, 8).transpose(1, 2)
        shifted = rope(x, torch.arange(4, 11))
        torch.testing.assert_close(shifted.reshape(2, 2, 7, 4, 2).square().sum(-1),
                                   x.reshape(2, 2, 7, 4, 2).square().sum(-1))
        # 用绝对位置：位置 4..10 的结果必须与位置 0..6 不同，否则等于根本没编码位置。
        self.assertFalse(torch.allclose(shifted, rope(x, torch.arange(7))))

    def test_rmsnorm_is_not_centered_layernorm(self):
        # 常量输入下 LayerNorm 会输出 0；RMSNorm 只按均方根缩放，输出仍是常数。
        # 这里手算期望值 2/sqrt(4+eps)，从数值上钉死"没有减均值"这一点。
        x = torch.full((1, 3, 16), 2.0, dtype=torch.float64)
        result = rms_norm(x, torch.ones(16), 1e-6)
        torch.testing.assert_close(result, torch.full_like(x, 2 / (4 + 1e-6)**0.5))

    def test_reject_bad_config_and_cache(self):
        # Config(15,3,24)：15/3=5 是奇数 head_dim，RoPE 无法配对，应在构造时就被拒绝。
        with self.assertRaises(ValueError):
            Config(15, 3, 24)
        # 空序列（S=0）应被拒绝，而不是产生 NaN 再往下传。
        with self.assertRaises(ValueError):
            self.model(self.x[:, :0])
        # cache 形状与本次 batch/heads 不符时应被拒绝，而不是静默广播出错误长度。
        cache = KVCache(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8))
        with self.assertRaises(ValueError):
            self.model(self.x, cache)


if __name__ == "__main__":
    unittest.main()
