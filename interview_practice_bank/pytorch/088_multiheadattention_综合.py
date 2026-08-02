"""
题目 088：MultiheadAttention_综合

要求：完成“MultiheadAttention”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入多头注意力层。
3. 保留每个head的Attention权重。
4. 综合验证head维和概率和。

完成标准：
- 综合验证head维和概率和。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入多头注意力层。
head_attention_module = nn.MultiheadAttention(8, 2, batch_first=True); head_tokens = torch.randn(2, 4, 8); head_output, per_head_weights = head_attention_module(head_tokens, head_tokens, head_tokens, need_weights=True, average_attn_weights=False)  # 保留每个head的Attention权重。
assert per_head_weights.shape == (2, 2, 4, 4) and torch.allclose(per_head_weights.sum(-1), torch.ones(2, 2, 4), atol=1e-5)  # 综合验证head维和概率和。
