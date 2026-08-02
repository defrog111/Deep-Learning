"""
题目 070：LSTM与隐藏状态_变式

要求：完成“LSTM与隐藏状态”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入LSTM模块。
3. LSTM直接处理PackedSequence。
4. 验证h和c状态。

完成标准：
- 验证h和c状态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入LSTM模块。
packed_lstm = nn.LSTM(3, 4, batch_first=True); lengths_lstm = torch.tensor([5, 3]); packed_input = nn.utils.rnn.pack_padded_sequence(torch.randn(2, 5, 3), lengths_lstm, batch_first=True, enforce_sorted=False); packed_output, (packed_h, packed_c) = packed_lstm(packed_input)  # LSTM直接处理PackedSequence。
assert packed_h.shape == packed_c.shape == (1, 2, 4)  # 验证h和c状态。
