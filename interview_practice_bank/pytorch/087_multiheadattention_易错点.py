"""
题目 087：MultiheadAttention_易错点

要求：完成“MultiheadAttention”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入多头注意力层。
3. 上三角布尔mask阻止查看未来token。
4. 第一个query只能关注自身。

完成标准：
- 第一个query只能关注自身。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入多头注意力层。
causal_attention_module = nn.MultiheadAttention(8, 2, batch_first=True); causal_tokens = torch.randn(2, 4, 8); causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1); causal_attention, causal_weights = causal_attention_module(causal_tokens, causal_tokens, causal_tokens, attn_mask=causal_mask, need_weights=True)  # 上三角布尔mask阻止查看未来token。
assert torch.count_nonzero(causal_weights[:, 0, 1:]) == 0  # 第一个query只能关注自身。
