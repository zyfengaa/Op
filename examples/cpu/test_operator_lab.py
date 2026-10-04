"""CPU 教学案例的边界与语义测试。运行：python -m unittest discover -s examples/cpu"""

import math
import unittest

from elementwise_lab import sigmoid_scalar, vector_add
from gemm_lab import naive_gemm, tiled_gemm
from indexing_lab import ceil_div, one_dimensional_grid, offset
from layernorm_lab import layer_norm
from reduction_lab import tree_max, tree_sum
from softmax_lab import online_normalizer, stable_softmax
from advanced_lab import attention, conv2d_single_channel, bias_relu


class OperatorLabTest(unittest.TestCase):
    def test_attention_causal_and_noncausal(self):
        q, k, v = [[1.0], [0.0]], [[1.0], [0.0]], [[10.0], [20.0]]
        full = attention(q, k, v)
        causal = attention(q, k, v, causal=True)
        self.assertAlmostEqual(full[0][0], 12.689414, places=5)
        self.assertEqual(causal, [[10.0], [15.0]])

    def test_conv_and_fusion(self):
        self.assertEqual(conv2d_single_channel(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]),
            [[-4.0, -4.0], [-4.0, -4.0]])
        self.assertEqual(bias_relu([[-2.0, 1.0], [3.0, -4.0]], [1.0, 2.0]),
                         [[0.0, 3.0], [4.0, 0.0]])

    def test_tail_and_stride(self):
        self.assertEqual(ceil_div(10, 4), 3)
        items = list(one_dimensional_grid(10, 4))
        self.assertEqual(len(items), 12)
        self.assertEqual(sum(valid for _, _, _, valid in items), 10)
        self.assertEqual(offset(2, 3, 8), 19)

    def test_elementwise_edge_values(self):
        self.assertEqual(vector_add([], []), [])
        self.assertEqual(vector_add([1.0, -2.0], [3.0, 4.0]), [4.0, 2.0])
        self.assertEqual(sigmoid_scalar(1000.0), 1.0)
        self.assertEqual(sigmoid_scalar(-1000.0), 0.0)

    def test_reduction_padding_identity(self):
        self.assertEqual(tree_sum([1.0, 2.0, 3.0, 4.0, 5.0])[0], 15.0)
        self.assertEqual(tree_max([-5.0, -2.0, -9.0]), -2.0)
        with self.assertRaises(ValueError):
            tree_max([])

    def test_softmax_large_values_and_online(self):
        x = [1000.0, 1001.0, 999.0, 998.0, 997.0]
        expected = stable_softmax(x)
        m, l = online_normalizer(x, block_size=2)
        actual = [math.exp(value - m) / l for value in x]
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(sum(actual), 1.0, places=12)

    def test_layernorm_constant_row(self):
        result = layer_norm([7.0] * 4, [1.0] * 4, [0.0] * 4)
        self.assertEqual(result, [0.0] * 4)

    def test_gemm_all_dimensions_have_tails(self):
        a = [[float((r + c) % 4) for c in range(5)] for r in range(3)]
        b = [[float((r - c) % 5) for c in range(4)] for r in range(5)]
        expected = naive_gemm(a, b)
        actual, _ = tiled_gemm(a, b, tile=2)
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            tiled_gemm(a, b, tile=0)


if __name__ == "__main__":
    unittest.main()
