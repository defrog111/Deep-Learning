"""Tiny Causal LM 的共享模型和字符 tokenizer。

训练入口与推理入口都导入这个文件，避免复制模型结构。
"""

from __future__ import annotations

import torch
from torch import nn

from components import EncoderBlock, SinusoidalPositionalEncoding, causal_mask


class CharTokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.token_to_id))
        if unknown:
            raise ValueError(f"prompt 含训练词表之外的字符：{unknown}")
        return [self.token_to_id[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.tokens[token_id] for token_id in token_ids)


class TinyCausalLM(nn.Module):
    """用带 causal mask 的 Encoder Block 构成 GPT 风格模型。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        block_size: int = 64,
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=block_size)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(d_model, num_heads, d_model * 4)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.size(1) > self.block_size:
            raise ValueError("输入长度超过 block_size。")
        x = self.position(self.embedding(token_ids))
        mask = causal_mask(token_ids.size(1), token_ids.device)
        for block in self.blocks:
            x, _ = block(x, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits = self(context)[:, -1]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids
