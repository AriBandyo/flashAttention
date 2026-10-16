import math
import unittest

import torch

from flashforge.reference import (
    attention_reference,
    causal_lower_right_mask,
    online_attention_reference,
    split_kv_decode_reference,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assertClose(self, a, b, atol=2e-5, rtol=2e-5):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def test_online_matches_exact_tail_tile(self):
        q = torch.randn(1, 2, 9, 64, dtype=torch.float32)
        k = torch.randn(1, 2, 73, 64, dtype=torch.float32)
        v = torch.randn(1, 2, 73, 64, dtype=torch.float32)
        ref = attention_reference(q, k, v)
        got = online_attention_reference(q, k, v, tile_k=32)
        self.assertClose(got, ref)

    def test_large_magnitude_logits_stay_finite(self):
        q = torch.randn(1, 1, 4, 64, dtype=torch.float64) * 1000.0
        k = torch.randn(1, 1, 11, 64, dtype=torch.float64) * 1000.0
        v = torch.randn(1, 1, 11, 64, dtype=torch.float64)
        ref = attention_reference(q, k, v)
        got = online_attention_reference(q, k, v, tile_k=5)
        self.assertTrue(torch.isfinite(got).all())
        self.assertClose(got, ref, atol=1e-10, rtol=1e-10)

    def test_fully_masked_row_is_zero_not_nan(self):
        q = torch.randn(1, 1, 3, 64)
        k = torch.randn(1, 1, 7, 64)
        v = torch.randn(1, 1, 7, 64)
        keep = torch.ones(1, 1, 3, 7, dtype=torch.bool)
        keep[:, :, 1, :] = False
        ref = attention_reference(q, k, v, keep_mask=keep)
        got = online_attention_reference(q, k, v, keep_mask=keep, tile_k=4)
        self.assertTrue(torch.equal(got[0, 0, 1], torch.zeros(64)))
        self.assertTrue(torch.isfinite(got).all())
        self.assertClose(got, ref)

    def test_causal_lower_right_q_shorter_than_k(self):
        mask = causal_lower_right_mask(2, 5)
        expected = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_causal_lower_right_q_longer_than_k_has_fully_masked_rows(self):
        mask = causal_lower_right_mask(5, 2)
        expected = torch.tensor([
            [False, False],
            [False, False],
            [False, False],
            [True, False],
            [True, True],
        ])
        self.assertTrue(torch.equal(mask, expected))
        q = torch.randn(1, 1, 5, 64)
        k = torch.randn(1, 1, 2, 64)
        v = torch.randn(1, 1, 2, 64)
        got = online_attention_reference(q, k, v, causal=True, tile_k=2)
        self.assertTrue(torch.equal(got[0, 0, :3], torch.zeros(3, 64)))
        self.assertTrue(torch.isfinite(got).all())

    def test_split_kv_decode_matches_exact(self):
        q = torch.randn(2, 3, 1, 128)
        k = torch.randn(2, 3, 257, 128)
        v = torch.randn(2, 3, 257, 128)
        ref = attention_reference(q, k, v)
        for splits in (1, 3, 8, 17):
            got = split_kv_decode_reference(q, k, v, splits=splits)
            self.assertClose(got, ref, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
