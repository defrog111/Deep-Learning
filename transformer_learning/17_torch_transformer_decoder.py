"""例 17：PyTorch 官方 nn.TransformerDecoder 的完整用法。

Decoder 接收两路输入：
1. tgt：已经生成/训练时右移后的目标 token。
2. memory：Encoder 的输出。
它先做因果 Self-Attention，再对 memory 做 Cross-Attention。
"""

import torch
from torch import nn


torch.manual_seed(0)
vocab_size, d_model = 30, 32
source_length, target_length = 7, 5

source_ids = torch.tensor(
    [
        [4, 8, 3, 2, 9, 1, 6],
        [7, 5, 2, 4, 0, 0, 0],
    ]
)
# 训练时 target_input 是答案右移一位后的输入，1 可视为 BOS。
target_input_ids = torch.tensor(
    [
        [1, 9, 6, 3, 8],
        [1, 4, 7, 0, 0],
    ]
)
target_label_ids = torch.tensor(
    [
        [9, 6, 3, 8, 2],
        [4, 7, 2, 0, 0],
    ]
)

source_padding_mask = source_ids.eq(0)  # (B, S)
target_padding_mask = target_input_ids.eq(0)  # (B, T)

embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
source_position = nn.Embedding(source_length, d_model)
target_position = nn.Embedding(target_length, d_model)

source = embedding(source_ids) + source_position(
    torch.arange(source_length).unsqueeze(0)
)
target = embedding(target_input_ids) + target_position(
    torch.arange(target_length).unsqueeze(0)
)

# 先用官方 Encoder 产生 memory。
encoder_layer = nn.TransformerEncoderLayer(
    d_model, nhead=4, dim_feedforward=128, batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
memory = encoder(source, src_key_padding_mask=source_padding_mask)

# 再构建并调用官方 Decoder。
decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=4,
    dim_feedforward=128,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=True,
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

# True 表示禁止关注；上三角屏蔽未来 token。使用 bool 与 padding mask 类型一致。
causal_mask = torch.ones(target_length, target_length, dtype=torch.bool).triu(1)
decoded = decoder(
    tgt=target,
    memory=memory,
    tgt_mask=causal_mask,
    tgt_key_padding_mask=target_padding_mask,
    memory_key_padding_mask=source_padding_mask,
)  # (B, T, D)

lm_head = nn.Linear(d_model, vocab_size)
logits = lm_head(decoded)
loss = nn.CrossEntropyLoss(ignore_index=0)(
    logits.reshape(-1, vocab_size),
    target_label_ids.reshape(-1),
)
loss.backward()

print("source / target ids:", source_ids.shape, target_input_ids.shape)
print("memory:", memory.shape)
print("causal mask:\n", causal_mask)
print("decoder output:", decoded.shape)
print("next-token logits:", logits.shape)
print("loss:", loss.item())
print(
    "decoder gradient exists:",
    decoder.layers[0].self_attn.in_proj_weight.grad is not None,
)
