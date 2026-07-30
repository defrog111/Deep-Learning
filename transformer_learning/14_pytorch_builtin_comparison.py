"""例 14：手写 Encoder Block 与 PyTorch 官方实现的接口对照。"""

import torch
from torch import nn

from components import EncoderBlock


torch.manual_seed(0)
x = torch.randn(2, 6, 32)

handwritten = EncoderBlock(32, num_heads=4, hidden_dim=128)
builtin = nn.TransformerEncoderLayer(
    d_model=32,
    nhead=4,
    dim_feedforward=128,
    batch_first=True,
    norm_first=True,
)

manual_output, manual_weights = handwritten(x)
builtin_output = builtin(x)

print("手写输出:", manual_output.shape)
print("官方输出:", builtin_output.shape)
print("手写版本额外暴露 attention weights:", manual_weights.shape)
print("手写参数量:", sum(p.numel() for p in handwritten.parameters()))
print("官方参数量:", sum(p.numel() for p in builtin.parameters()))
print("注意：随机初始化不同，因此这里只比较结构与 shape，不比较数值。")
