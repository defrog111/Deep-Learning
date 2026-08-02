"""
题目 086：MultiheadAttention_变式

要求：完成“MultiheadAttention”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入多头注意力层。
3. key_padding_mask按batch屏蔽key位置。
4. 验证batch_first输出与权重。

完成标准：
- 验证batch_first输出与权重。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入多头注意力层。
batch_attention = nn.MultiheadAttention(8, 2, batch_first=True); attention_tokens = torch.randn(2, 4, 8); padding_mask = torch.tensor([[False, False, True, True], [False, False, False, True]]); masked_attention, masked_weights = batch_attention(attention_tokens, attention_tokens, attention_tokens, key_padding_mask=padding_mask, need_weights=True)  # key_padding_mask按batch屏蔽key位置。
assert masked_attention.shape == attention_tokens.shape and masked_weights.shape == (2, 4, 4)  # 验证batch_first输出与权重。
