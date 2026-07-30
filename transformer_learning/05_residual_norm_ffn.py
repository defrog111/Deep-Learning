"""例 05：LayerNorm、残差连接和逐 token Feed Forward。"""

import torch
from torch import nn

from components import FeedForward


torch.manual_seed(0)
x = torch.randn(2, 5, 16)
norm = nn.LayerNorm(16)
ffn = FeedForward(d_model=16, hidden_dim=64)
output = x + ffn(norm(x))

print("input/output:", x.shape, output.shape)
print("归一化后最后一维均值:", norm(x).mean(dim=-1))
print("残差使梯度能沿加法路径传播。")
