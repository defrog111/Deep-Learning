"""
题目 088：MultiheadAttention_综合

要求：完成“MultiheadAttention”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建batch-first多头注意力。
3. 创建token表示。
4. True表示屏蔽key。
5. 执行Self-Attention。
6. 保留每个head的Attention权重。
7. 综合验证head维和概率和。

完成标准：
- 验证输出和逐头权重shape。
- 综合验证head维和概率和。
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
head_attention_module = nn.MultiheadAttention(8, 2, batch_first=True); head_tokens = torch.randn(2, 4, 8); head_output, per_head_weights = head_attention_module(head_tokens, head_tokens, head_tokens, need_weights=True, average_attn_weights=False)  # 保留每个head的Attention权重。
assert per_head_weights.shape == (2, 2, 4, 4) and torch.allclose(per_head_weights.sum(-1), torch.ones(2, 2, 4), atol=1e-5)  # 综合验证head维和概率和。
