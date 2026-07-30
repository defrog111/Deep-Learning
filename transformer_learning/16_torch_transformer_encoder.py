"""例 16：PyTorch 官方 nn.TransformerEncoder 的完整用法。

Encoder 常用于 BERT 风格编码、文本分类、序列分类和特征提取。
输入采用 batch_first=True，因此 shape 为 (batch, sequence, d_model)。
"""

import torch
from torch import nn


torch.manual_seed(0)
batch_size, sequence_length, d_model = 3, 6, 32

# 1. 原始 token id；0 作为 padding token。
token_ids = torch.tensor(
    [
        [5, 8, 2, 9, 4, 1],
        [7, 3, 6, 2, 0, 0],
        [4, 9, 1, 0, 0, 0],
    ]
)
padding_mask = token_ids.eq(0)  # (B, T)，官方 API 中 True 表示要屏蔽。

# 2. token embedding + 可学习位置 embedding。
token_embedding = nn.Embedding(20, d_model, padding_idx=0)
position_embedding = nn.Embedding(sequence_length, d_model)
positions = torch.arange(sequence_length).unsqueeze(0)
encoder_input = token_embedding(token_ids) + position_embedding(positions)

# 3. 一层的结构配置，然后堆叠为多层 Encoder。
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=4,
    dim_feedforward=d_model * 4,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=True,
)
encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=2,
    norm=nn.LayerNorm(d_model),
    enable_nested_tensor=False,
)

# 4. src_key_padding_mask 防止模型关注 padding key。
encoded = encoder(
    src=encoder_input,
    src_key_padding_mask=padding_mask,
)  # (B, T, D)

# 5. 示例任务：masked mean pooling 后进行三分类。
valid_tokens = (~padding_mask).unsqueeze(-1)
pooled = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
classifier = nn.Linear(d_model, 3)
logits = classifier(pooled)
labels = torch.tensor([0, 1, 2])
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()

print("token ids:", token_ids.shape)
print("padding mask (True means ignored):\n", padding_mask)
print("encoder input:", encoder_input.shape)
print("encoder output:", encoded.shape)
print("classification logits:", logits.shape)
print("loss:", loss.item())
print("encoder gradient exists:", encoder.layers[0].self_attn.in_proj_weight.grad is not None)
