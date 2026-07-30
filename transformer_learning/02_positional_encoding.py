"""例 02：正弦位置编码，让 Attention 能区分 token 顺序。"""

import torch

from components import SinusoidalPositionalEncoding


x = torch.zeros(1, 6, 8)
position = SinusoidalPositionalEncoding(d_model=8, max_len=32)
y = position(x)

print("output shape:", y.shape)
print("position 0:", y[0, 0])
print("position 1:", y[0, 1])
print("位置不同:", not torch.allclose(y[0, 0], y[0, 1]))
