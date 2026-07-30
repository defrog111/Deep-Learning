"""例 04：把特征拆成多个 head，并行关注不同关系。"""

import torch

from components import MultiHeadAttention


torch.manual_seed(0)
x = torch.randn(2, 5, 16)
attention = MultiHeadAttention(d_model=16, num_heads=4)
output, weights = attention(x)

print("input/output:", x.shape, output.shape)
print("weights (B, H, T, T):", weights.shape)
print("head 0 的第一个 query 权重:", weights[0, 0, 0])
