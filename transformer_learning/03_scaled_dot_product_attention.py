"""例 03：单头 scaled dot-product attention 的完整计算。"""

import torch

from components import scaled_dot_product_attention


torch.manual_seed(0)
query = torch.randn(1, 1, 3, 4)
key = torch.randn(1, 1, 3, 4)
value = torch.randn(1, 1, 3, 4)
context, weights = scaled_dot_product_attention(query, key, value)

print("attention weights shape:", weights.shape)
print("每行权重和:", weights.sum(dim=-1))
print("context shape:", context.shape)
