"""例 08：Decoder 的 masked self-attention 和 cross-attention。"""

import torch

from components import DecoderBlock, causal_mask


torch.manual_seed(0)
memory = torch.randn(2, 7, 32)  # Encoder 输出
target = torch.randn(2, 5, 32)  # Decoder 当前输入
block = DecoderBlock(d_model=32, num_heads=4, hidden_dim=128)
output, self_weights, cross_weights = block(
    target, memory, target_mask=causal_mask(target.size(1))
)

print("decoder output:", output.shape)
print("self-attention:", self_weights.shape)
print("cross-attention:", cross_weights.shape)
