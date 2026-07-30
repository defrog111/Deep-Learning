"""可复用的手写 Transformer 组件。

所有核心计算都使用基础 PyTorch 张量运算实现，便于逐行理解。
张量约定统一为 batch-first: (batch, sequence, d_model)。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V。

    query/key/value: (B, H, T, D)
    mask: 可广播到 (B, H, T_query, T_key)，True 表示允许关注。
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ value, weights


class SinusoidalPositionalEncoding(nn.Module):
    """论文中的正弦/余弦位置编码，无需学习参数。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model 必须为偶数，才能配对 sin/cos。")
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    """从零实现多头注意力，支持 self-attention 和 cross-attention。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除。")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = query if key is None else key
        value = key if value is None else value
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        batch, _, seq_len, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(context), weights


class FeedForward(nn.Module):
    """逐 token 的两层 MLP：D -> 4D -> D。"""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-LN Encoder Block：Self-Attention + FFN，各带残差连接。"""

    def __init__(
        self, d_model: int, num_heads: int, hidden_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, weights = self.attention(
            self.norm1(x), mask=mask
        )
        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, weights


class DecoderBlock(nn.Module):
    """Decoder Block：因果 Self-Attention + Cross-Attention + FFN。"""

    def __init__(
        self, d_model: int, num_heads: int, hidden_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm1(target)
        self_output, self_weights = self.self_attention(
            normalized, mask=target_mask
        )
        target = target + self.dropout(self_output)
        cross_output, cross_weights = self.cross_attention(
            self.norm2(target), key=memory, value=memory, mask=memory_mask
        )
        target = target + self.dropout(cross_output)
        target = target + self.dropout(self.feed_forward(self.norm3(target)))
        return target, self_weights, cross_weights


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """返回 (1, 1, T, T) 的下三角布尔 mask。"""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril().view(
        1, 1, length, length
    )


def padding_mask(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """把 (B, T) token ids 转成可广播的 (B, 1, 1, T) mask。"""
    return token_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)
