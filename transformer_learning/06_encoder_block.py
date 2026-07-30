"""例 06：完整 Transformer Encoder Block。"""

import torch

from components import EncoderBlock


torch.manual_seed(0)
x = torch.randn(2, 6, 32)
block = EncoderBlock(d_model=32, num_heads=4, hidden_dim=128)
output, attention_weights = block(x)
loss = output.square().mean()
loss.backward()

print("output:", output.shape)
print("attention:", attention_weights.shape)
print("q_proj gradient norm:", block.attention.q_proj.weight.grad.norm().item())
