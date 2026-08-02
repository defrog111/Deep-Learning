"""
题目 064：Embedding与padding_idx_综合

要求：完成“Embedding与padding_idx”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Embedding模块。
3. 综合Embedding、padding和pack。
4. pack后只保留三个有效token。

完成标准：
- pack后只保留三个有效token。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Embedding模块。
token_ids = torch.tensor([[1, 2, 0], [3, 0, 0]]); lengths = torch.tensor([2, 1]); embedded_tokens = nn.Embedding(5, 4, padding_idx=0)(token_ids); packed_tokens = nn.utils.rnn.pack_padded_sequence(embedded_tokens, lengths, batch_first=True, enforce_sorted=False)  # 综合Embedding、padding和pack。
assert packed_tokens.data.shape == (3, 4)  # pack后只保留三个有效token。
