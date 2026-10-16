import unittest

from flashforge.roofline import (
    active_attention_pairs,
    attention_flops,
    ideal_fused_bytes,
    naive_materialized_bytes,
    roofline_metrics,
)


class RooflineTests(unittest.TestCase):
    def test_prefill_flop_formula(self):
        self.assertEqual(attention_flops(2, 8, 512, 512, 64, False), 4 * 2 * 8 * 512 * 512 * 64)

    def test_equal_length_causal_pairs(self):
        self.assertEqual(active_attention_pairs(8, 8, True), 8 * 9 // 2)

    def test_lower_right_unequal_pairs(self):
        self.assertEqual(active_attention_pairs(2, 5, True), 4 + 5)
        self.assertEqual(active_attention_pairs(5, 2, True), 0 + 0 + 0 + 1 + 2)

    def test_naive_moves_more_than_fused(self):
        fused = ideal_fused_bytes(1, 1, 1024, 1024, 64, 2)
        naive = naive_materialized_bytes(1, 1, 1024, 1024, 64, 2)
        self.assertGreater(naive, fused)

    def test_roofline_uses_minimum_ceiling(self):
        m = roofline_metrics(flops=10_000, bytes_moved=10_000, compute_tflops=100.0, bandwidth_gbs=1000.0)
        self.assertAlmostEqual(m.arithmetic_intensity, 1.0)
        self.assertAlmostEqual(m.roofline_tflops, 1.0)


if __name__ == "__main__":
    unittest.main()
