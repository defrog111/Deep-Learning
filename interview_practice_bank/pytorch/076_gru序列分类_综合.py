"""
题目 076：GRU序列分类_综合

要求：完成“GRU序列分类”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入GRU和线性层。
3. 综合变长序列GRU。
4. 验证有效步数和状态shape。

完成标准：
- 验证有效步数和状态shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入GRU和线性层。
packed_gru = nn.GRU(3, 4, batch_first=True); gru_lengths = torch.tensor([4, 2]); gru_packed = nn.utils.rnn.pack_padded_sequence(torch.randn(2, 4, 3), gru_lengths, batch_first=True, enforce_sorted=False); gru_packed_output, gru_hidden = packed_gru(gru_packed)  # 综合变长序列GRU。
assert gru_packed_output.data.shape[0] == 6 and gru_hidden.shape == (1, 2, 4)  # 验证有效步数和状态shape。
