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
        # threads 取 8 > len(row)=5：后面的 lane 分不到元素（空转线程），
        # 用来验证空 lane 不会污染 max/sum。covered 必须恰好是 0..4 各一次，
        # 即没有元素被漏掉、也没有元素被两个 lane 重复处理（对应 RD01）。
        row = [1000.0, 1001.0, 999.0, 998.0, 1002.0]
        for threads in (1, 2, 4, 8):
            with self.subTest(threads=threads):
                result = softmax_threads(row, threads)
                covered = sorted(i for lane in result["ownership"] for i in lane)
                self.assertEqual(covered, list(range(len(row))))
                # 结果不能随"线程数"改变——这是划分不变性，也是数据流模拟的判据。
                for actual, expected in zip(result["output"], stable_softmax(row)):
                    self.assertAlmostEqual(actual, expected, places=13)
        # 固定 stride 布局：lane0 拿走下标 0 和 4。
        self.assertEqual(softmax_threads(row, 4)["ownership"][0], [0, 4])

    def test_bad_zero_max_can_lose_stability_without_always_changing_result(self):
        # 展示 SM01 的两种结局：普通负数即使不减去 max 也恰好正确（因为 exp 没溢出），
        # 所以这类 bug 会被"普通输入"掩盖；只有极负值才暴露。
        ordinary = [-5.0, -4.0, -3.0]
        shifted_by_zero = [math.exp(x) for x in ordinary]
        denominator = sum(shifted_by_zero)
        for actual, expected in zip(shifted_by_zero, stable_softmax(ordinary)):
            self.assertAlmostEqual(actual / denominator, expected, places=13)
        # [-1000,-1001] 直接 exp 全部下溢为 0，分母为 0 → 失败；
        # 先减 max 后 exp(-0)/(-1) 得到 (1, e^-1)，归一化后和为 1。
        self.assertEqual(sum(math.exp(x) for x in [-1000.0, -1001.0]), 0.0)
        self.assertAlmostEqual(sum(stable_softmax([-1000.0, -1001.0])), 1.0)

    def test_welford_updates_and_unequal_partition_merge(self):
        # 遍历所有切分点（含 split=0 与 split=4，即一侧为空）：验证空分区合并的
        # identity 正确，且合并结果与如何切分无关（对应 LN01 的带 count 合并）。
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(welford_state(x), (4, 2.5, 5.0))
        for split in range(5):
            merged = merge_welford(welford_state(x[:split]), welford_state(x[split:]))
            self.assertEqual(merged[0], 4)
            self.assertAlmostEqual(merged[1], 2.5)
            self.assertAlmostEqual(merged[2], 5.0)

    def test_welford_constant_and_offset_inputs(self):
        # 常数序列的 M2 必须是精确 0，不能是一堆 1e-12 噪声的累加。
        self.assertEqual(welford_state([10000.0] * 7), (7, 10000.0, 0.0))
        # 10000 附近、彼此相差 1 的数：两遍法在这里会损失精度，Welford 不会。
        x = [10001.0, 10002.0, 10003.0, 10004.0]
        self.assertEqual(welford_state(x), (4, 10002.5, 5.0))

    def test_gemm_trace_preserves_accumulation_and_zero_padding(self):
        first, second = gemm_first_tile_trace()
        # 第一个 K tile 之后的累加器。
        self.assertEqual(first["accumulators"], [[25.0, 28.0], [73.0, 82.0]])
        # 第二个 K tile 的 A 片段：k=2 之后越界的位置被补 0（[3,0] 而不是越界读取）。
        self.assertEqual(second["a_tile"], [[3.0, 0.0], [6.0, 0.0]])
        # 跨 tile 累加必须继续叠加，而不是覆盖。
        self.assertEqual(second["accumulators"], [[58.0, 64.0], [139.0, 154.0]])

    def test_attention_output_rescales_with_new_maximum(self):
        actual, traces = blocked_attention([[1.0]], [[0.0], [2.0]], [[10.0], [20.0]], 1)
        expected = (10.0 + 20.0 * math.exp(2.0)) / (1.0 + math.exp(2.0))
        self.assertAlmostEqual(actual[0][0], expected)
        # 遇到新的 max 后，旧的 o 被缩放：10*exp(-2)+20 对应重缩放因子 alpha=exp(-2)。
        self.assertAlmostEqual(traces[0][-1]["o"][0], 10 * math.exp(-2) + 20)
        # 注意力输出是 V 的凸组合，必须落在可见 V 的 [10,20] 区间内（AT01）。
        self.assertTrue(10.0 <= actual[0][0] <= 20.0)

    def test_attention_tail_blocks_causal_and_partition_invariance(self):
        # 随机输入 + 多种 block：block=3、8 都不能整除 len(k)=5，会走到尾块；
        # causal 与非 causal 都要与朴素两遍实现一致——分块和遮罩不能改变结果。
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
        # 非方形 kernel（2 行 3 列）最容易暴露拍平顺序写反的错误。
        kernel = [[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]]
        actual, windows = im2col_single_channel(x, kernel)
        self.assertEqual(actual, conv2d_single_channel(x, kernel))
        # 第一个窗口是输入左上角 2×3 区域，按 ky 外层、kx 内层拍平。
        self.assertEqual(windows[0], [0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
        # 输出位置数 = (h-kh+1)*(w-kw+1) = 2*2 = 4。
        self.assertEqual(len(windows), 4)

    def test_reject_invalid_inputs(self):
        # 四条错误路径必须 raise，而不是静默算出错误结果：
        # 空行、Inf 输入、block_size=0（除零）、kernel 比输入还大。
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
