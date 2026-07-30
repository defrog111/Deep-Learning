"""例 09：从 Encoder/Decoder Block 组装完整 Seq2Seq Transformer。"""

import torch
from torch import nn

from components import (
    DecoderBlock,
    EncoderBlock,
    SinusoidalPositionalEncoding,
    causal_mask,
)


class TinySeq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        self.encoders = nn.ModuleList(
            [EncoderBlock(d_model, 4, d_model * 4) for _ in range(2)]
        )
        self.decoders = nn.ModuleList(
            [DecoderBlock(d_model, 4, d_model * 4) for _ in range(2)]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self, source_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        memory = self.position(self.embedding(source_ids))
        for layer in self.encoders:
            memory, _ = layer(memory)
        target = self.position(self.embedding(target_ids))
        mask = causal_mask(target_ids.size(1), target_ids.device)
        for layer in self.decoders:
            target, _, _ = layer(target, memory, target_mask=mask)
        return self.output(target)


torch.manual_seed(0)
model = TinySeq2SeqTransformer(vocab_size=100)
source = torch.randint(0, 100, (2, 7))
target_input = torch.randint(0, 100, (2, 5))
logits = model(source, target_input)
print("source:", source.shape)
print("target input:", target_input.shape)
print("logits (B, T_target, vocab):", logits.shape)
