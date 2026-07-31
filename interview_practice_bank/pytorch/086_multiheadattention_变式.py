"""
题目 086：MultiheadAttention_变式

要求：完成“MultiheadAttention”的变式题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入多头注意力层。
torch.manual_seed(0)  # 固定随机种子。
attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)  # 创建batch-first多头注意力。
tokens = torch.randn(2, 5, 16)  # 创建token表示。
padding_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])  # True表示屏蔽key。
output, weights = attention(tokens, tokens, tokens, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)  # 执行Self-Attention。
assert output.shape == tokens.shape and weights.shape == (2, 4, 5, 5)  # 验证输出和逐头权重shape。
print(output.shape, weights[0, 0])  # 输出shape和一个head权重。
