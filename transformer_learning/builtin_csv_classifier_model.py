"""PyTorch 原生多层 Transformer CSV 分类器的共享模型。

训练和推理脚本共同导入本文件，保证两边使用完全相同的网络结构。
"""

from __future__ import annotations

import torch
from torch import nn


class NumericFeatureTokenizer(nn.Module):
    """把一行中的每个数值特征变成一个 d_model 维 token。"""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.feature_embedding = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, F, 1)；每个 scalar 独立投影成一个 token。
        return (
            features.unsqueeze(-1) * self.scale
            + self.bias
            + self.feature_embedding
        )


class BuiltinTabularTransformerClassifier(nn.Module):
    """用 nn.TransformerEncoder 构建的多层表格分类器。"""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 48,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能够被 num_heads 整除。")

        self.num_features = num_features
        self.tokenizer = NumericFeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 这是 PyTorch 官方层，不是本项目手写的 EncoderBlock。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # PyTorch 会复制 encoder_layer，形成 num_layers 个参数互不共享的层。
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        feature_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.num_features:
            raise ValueError(
                f"features 应为 (batch, {self.num_features})，实际为 {tuple(features.shape)}"
            )

        feature_tokens = self.tokenizer(features)
        cls = self.cls_token.expand(features.size(0), -1, -1)
        tokens = torch.cat([cls, feature_tokens], dim=1)

        padding_mask = None
        if feature_padding_mask is not None:
            if feature_padding_mask.shape != features.shape:
                raise ValueError("feature_padding_mask 的 shape 必须与 features 相同。")
            # [CLS] 永远不是 padding；True 表示该位置不允许被 Attention 看见。
            cls_is_not_padding = torch.zeros(
                features.size(0), 1, dtype=torch.bool, device=features.device
            )
            padding_mask = torch.cat(
                [cls_is_not_padding, feature_padding_mask.bool()], dim=1
            )

        encoded = self.encoder(
            src=tokens,
            src_key_padding_mask=padding_mask,
        )
        # 第 0 个 token 是 [CLS]，汇总整行特征后用于分类。
        return self.classifier(encoded[:, 0])
