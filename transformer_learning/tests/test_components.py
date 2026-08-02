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
from builtin_causal_lm_model import BuiltinCausalLM  # noqa: E402
from builtin_csv_classifier_model import (  # noqa: E402
    BuiltinTabularTransformerClassifier,
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

    def test_builtin_csv_classifier_is_multilayer(self):
        model = BuiltinTabularTransformerClassifier(
            num_features=4,
            num_classes=3,
            d_model=16,
            num_heads=4,
            num_layers=3,
            dim_feedforward=32,
            dropout=0.0,
        )
        features = torch.randn(2, 4)
        feature_padding_mask = torch.tensor(
            [[False, False, False, False], [False, False, False, True]]
        )
        logits = model(features, feature_padding_mask)
        self.assertEqual(len(model.encoder.layers), 3)
        self.assertEqual(logits.shape, (2, 3))

    def test_builtin_causal_lm_is_multilayer_and_causal(self):
        model = BuiltinCausalLM(
            vocab_size=10,
            d_model=16,
            num_heads=4,
            num_layers=3,
            dim_feedforward=32,
            block_size=8,
            dropout=0.0,
        ).eval()
        original = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 8, 9]])
        with torch.no_grad():
            original_logits = model(original)
            changed_logits = model(changed_future)
        self.assertEqual(len(model.transformer.layers), 3)
        self.assertEqual(original_logits.shape, (1, 4, 10))
        # 第 0、1 个位置不能看到第 2、3 个未来 token。
        self.assertTrue(
            torch.allclose(
                original_logits[:, :2], changed_logits[:, :2], atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
