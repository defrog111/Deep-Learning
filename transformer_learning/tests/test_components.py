"""核心组件的 shape、mask、梯度和概率性质测试。"""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1]))
from components import (  # noqa: E402
    DecoderBlock,
    EncoderBlock,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
)


class TransformerComponentTests(unittest.TestCase):
    def test_attention_probabilities_sum_to_one(self):
        q = k = v = torch.randn(2, 3, 5, 4)
        output, weights = scaled_dot_product_attention(q, k, v)
        self.assertEqual(output.shape, (2, 3, 5, 4))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 3, 5)))

    def test_causal_mask_blocks_future(self):
        mask = causal_mask(4)[0, 0]
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_padding_mask_shape(self):
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
        mask = padding_mask(ids)
        self.assertEqual(mask.shape, (2, 1, 1, 3))
        self.assertFalse(mask[0, 0, 0, 2])

    def test_positional_encoding_changes_positions(self):
        output = SinusoidalPositionalEncoding(8)(torch.zeros(1, 3, 8))
        self.assertFalse(torch.allclose(output[:, 0], output[:, 1]))

    def test_multihead_shape_and_gradient(self):
        layer = MultiHeadAttention(16, 4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        output, weights = layer(x)
        output.sum().backward()
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        self.assertIsNotNone(x.grad)

    def test_encoder_and_decoder_shapes(self):
        encoder = EncoderBlock(16, 4, 64)
        decoder = DecoderBlock(16, 4, 64)
        source, _ = encoder(torch.randn(2, 7, 16))
        target, self_weights, cross_weights = decoder(
            torch.randn(2, 5, 16), source, causal_mask(5)
        )
        self.assertEqual(target.shape, (2, 5, 16))
        self.assertEqual(self_weights.shape, (2, 4, 5, 5))
        self.assertEqual(cross_weights.shape, (2, 4, 5, 7))


if __name__ == "__main__":
    unittest.main()
