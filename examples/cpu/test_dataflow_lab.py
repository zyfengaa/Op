"""Check the intermediate-state lessons against independent semantic references."""

import math
import random
import unittest

from advanced_lab import attention, conv2d_single_channel
from dataflow_lab import (
    blocked_attention, gemm_first_tile_trace, im2col_single_channel,
    merge_welford, softmax_threads, welford_state,
)
from softmax_lab import stable_softmax


class DataflowLabTest(unittest.TestCase):
    def test_lane_ownership_and_idle_threads(self):
        row = [1000.0, 1001.0, 999.0, 998.0, 1002.0]
        for threads in (1, 2, 4, 8):
            with self.subTest(threads=threads):
                result = softmax_threads(row, threads)
                covered = sorted(i for lane in result["ownership"] for i in lane)
                self.assertEqual(covered, list(range(len(row))))
                for actual, expected in zip(result["output"], stable_softmax(row)):
                    self.assertAlmostEqual(actual, expected, places=13)
        self.assertEqual(softmax_threads(row, 4)["ownership"][0], [0, 4])

    def test_bad_zero_max_can_lose_stability_without_always_changing_result(self):
        ordinary = [-5.0, -4.0, -3.0]
        shifted_by_zero = [math.exp(x) for x in ordinary]
        denominator = sum(shifted_by_zero)
        for actual, expected in zip(shifted_by_zero, stable_softmax(ordinary)):
            self.assertAlmostEqual(actual / denominator, expected, places=13)
        self.assertEqual(sum(math.exp(x) for x in [-1000.0, -1001.0]), 0.0)
        self.assertAlmostEqual(sum(stable_softmax([-1000.0, -1001.0])), 1.0)

    def test_welford_updates_and_unequal_partition_merge(self):
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(welford_state(x), (4, 2.5, 5.0))
        for split in range(5):
            merged = merge_welford(welford_state(x[:split]), welford_state(x[split:]))
            self.assertEqual(merged[0], 4)
            self.assertAlmostEqual(merged[1], 2.5)
            self.assertAlmostEqual(merged[2], 5.0)

    def test_welford_constant_and_offset_inputs(self):
        self.assertEqual(welford_state([10000.0] * 7), (7, 10000.0, 0.0))
        x = [10001.0, 10002.0, 10003.0, 10004.0]
        self.assertEqual(welford_state(x), (4, 10002.5, 5.0))

    def test_gemm_trace_preserves_accumulation_and_zero_padding(self):
        first, second = gemm_first_tile_trace()
        self.assertEqual(first["accumulators"], [[25.0, 28.0], [73.0, 82.0]])
        self.assertEqual(second["a_tile"], [[3.0, 0.0], [6.0, 0.0]])
        self.assertEqual(second["accumulators"], [[58.0, 64.0], [139.0, 154.0]])

    def test_attention_output_rescales_with_new_maximum(self):
        actual, traces = blocked_attention([[1.0]], [[0.0], [2.0]], [[10.0], [20.0]], 1)
        expected = (10.0 + 20.0 * math.exp(2.0)) / (1.0 + math.exp(2.0))
        self.assertAlmostEqual(actual[0][0], expected)
        self.assertAlmostEqual(traces[0][-1]["o"][0], 10 * math.exp(-2) + 20)
        self.assertTrue(10.0 <= actual[0][0] <= 20.0)

    def test_attention_tail_blocks_causal_and_partition_invariance(self):
        rng = random.Random(17)
        q = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(4)]
        k = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(5)]
        v = [[rng.uniform(-2, 2) for _ in range(2)] for _ in range(5)]
        for causal in (False, True):
            expected = attention(q, k, v, causal=causal)
            for block in (1, 2, 3, 8):
                with self.subTest(causal=causal, block=block):
                    actual, _ = blocked_attention(q, k, v, block, causal)
                    for ar, er in zip(actual, expected):
                        for a, e in zip(ar, er):
                            self.assertAlmostEqual(a, e, places=12)

    def test_im2col_matches_direct_for_asymmetric_window(self):
        x = [[float(r * 4 + c) for c in range(4)] for r in range(3)]
        kernel = [[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]]
        actual, windows = im2col_single_channel(x, kernel)
        self.assertEqual(actual, conv2d_single_channel(x, kernel))
        self.assertEqual(windows[0], [0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
        self.assertEqual(len(windows), 4)

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            softmax_threads([], 4)
        with self.assertRaises(ValueError):
            welford_state([math.inf])
        with self.assertRaises(ValueError):
            blocked_attention([[1.0]], [[1.0]], [[1.0]], 0)
        with self.assertRaises(ValueError):
            im2col_single_channel([[1]], [[1, 2]])


if __name__ == "__main__":
    unittest.main()
