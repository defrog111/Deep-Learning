"""
题目 086：MultiheadAttention_变式

要求：完成“MultiheadAttention”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建batch-first多头注意力。
3. 创建token表示。
4. True表示屏蔽key。
5. 执行Self-Attention。
6. key_padding_mask按batch屏蔽key位置。

完成标准：
- 验证输出和逐头权重shape。
- 验证batch_first输出与权重。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
batch_attention = nn.MultiheadAttention(8, 2, batch_first=True); attention_tokens = torch.randn(2, 4, 8); padding_mask = torch.tensor([[False, False, True, True], [False, False, False, True]]); masked_attention, masked_weights = batch_attention(attention_tokens, attention_tokens, attention_tokens, key_padding_mask=padding_mask, need_weights=True)  # key_padding_mask按batch屏蔽key位置。
assert masked_attention.shape == attention_tokens.shape and masked_weights.shape == (2, 4, 4)  # 验证batch_first输出与权重。
