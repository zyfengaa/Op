import unittest
import torch
from block import Config, KVCache, TinyDecoderBlock, rms_norm, rope


class BlockTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = TinyDecoderBlock(Config(16, 2, 24), dtype=torch.float64)
        g = torch.Generator().manual_seed(11)
        self.x = torch.randn(2, 7, 16, generator=g, dtype=torch.float64)

    def test_packed_sdpa_matches_explicit(self):
        with torch.inference_mode():
            a, _ = self.model(self.x, variant="reference")
            b, _ = self.model(self.x, variant="optimized")
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_prefill_then_decode_matches_full_and_does_not_mutate_cache(self):
        with torch.inference_mode():
            for variant in ("reference", "optimized"):
                full, _ = self.model(self.x, variant=variant)
                _, cache = self.model(self.x[:, :4], variant=variant)
                saved = cache.k.clone()
                suffix, extended = self.model(self.x[:, 4:], cache, variant)
                torch.testing.assert_close(suffix, full[:, 4:], atol=1e-10, rtol=1e-10)
                torch.testing.assert_close(cache.k, saved, atol=0, rtol=0)
                self.assertEqual(cache.length, 4)
                self.assertEqual(extended.length, 7)

    def test_every_token_incremental_matches_full(self):
        with torch.inference_mode():
            full, _ = self.model(self.x, variant="optimized")
            cache, pieces = None, []
            for token in range(self.x.shape[1]):
                output, cache = self.model(self.x[:, token:token + 1], cache, "optimized")
                pieces.append(output)
            torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-10, rtol=1e-10)

    def test_future_tokens_do_not_change_past_outputs(self):
        changed = self.x.clone()
        changed[:, 4:] += 100
        with torch.inference_mode():
            for variant in ("reference", "optimized"):
                first, _ = self.model(self.x, variant=variant)
                second, _ = self.model(changed, variant=variant)
                torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-10, rtol=1e-10)

    def test_rope_preserves_pair_norm_and_uses_absolute_position(self):
        x = self.x.reshape(2, 7, 2, 8).transpose(1, 2)
        shifted = rope(x, torch.arange(4, 11))
        torch.testing.assert_close(shifted.reshape(2, 2, 7, 4, 2).square().sum(-1),
                                   x.reshape(2, 2, 7, 4, 2).square().sum(-1))
        self.assertFalse(torch.allclose(shifted, rope(x, torch.arange(7))))

    def test_rmsnorm_is_not_centered_layernorm(self):
        x = torch.full((1, 3, 16), 2.0, dtype=torch.float64)
        result = rms_norm(x, torch.ones(16), 1e-6)
        torch.testing.assert_close(result, torch.full_like(x, 2 / (4 + 1e-6)**0.5))

    def test_reject_bad_config_and_cache(self):
        with self.assertRaises(ValueError):
            Config(15, 3, 24)
        with self.assertRaises(ValueError):
            self.model(self.x[:, :0])
        cache = KVCache(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8))
        with self.assertRaises(ValueError):
            self.model(self.x, cache)


if __name__ == "__main__":
    unittest.main()
