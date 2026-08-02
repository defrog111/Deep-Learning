"""PyTorch 原生多层 causal language model 的共享模型。

GPT 属于 decoder-only Transformer：只有 masked self-attention，没有 encoder
memory/cross-attention。因此这里使用官方 nn.TransformerEncoder，并给每一层传入
causal mask。模块名叫 Encoder，但加上 causal mask 后数据流就是 GPT 式 decoder-only。
"""

from __future__ import annotations

import torch
from torch import nn


class CharTokenizer:
    """教学用字符 tokenizer；真实 LLM 通常使用 BPE/SentencePiece。"""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.token_to_id))
        if unknown:
            raise ValueError(f"prompt 含词表外字符：{unknown}")
        return [self.token_to_id[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.tokens[token_id] for token_id in token_ids)


class BuiltinCausalLM(nn.Module):
    """Token/Position Embedding + 多层官方 Transformer + LM Head。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 192,
        block_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能够被 num_heads 整除。")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 小标准差初始化让未训练 logits 不会过大，Tiny LM 的 loss 更稳定。
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        # 输入 Embedding 与输出分类矩阵共享权重，是 LLM 中常见的 weight tying。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """返回 (T,T) bool mask；True 表示不允许关注未来 token。"""
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 (batch, sequence) 二维张量。")
        batch_size, length = token_ids.shape
        if length > self.block_size:
            raise ValueError("输入长度超过 block_size。")
        if padding_mask is not None and padding_mask.shape != token_ids.shape:
            raise ValueError("padding_mask 的 shape 必须与 token_ids 相同。")

        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = self.make_causal_mask(length, token_ids.device)
        hidden = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        # 每个位置输出整个词表的 logits：(B,T,D) -> (B,T,V)。
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """逐 token 自回归生成；训练和 inference 的核心差别就在这里。"""
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            next_token_logits = self(context)[:, -1]
            if temperature <= 0:
                next_id = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(
                    next_token_logits / temperature, dim=-1
                )
                next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids
