"""
题目 072：LSTM与隐藏状态_综合

要求：完成“LSTM与隐藏状态”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入LSTM模块。
3. 综合多层双向LSTM。
4. 层和方向合并在状态首维。

完成标准：
- 层和方向合并在状态首维。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入LSTM模块。
bi_lstm = nn.LSTM(3, 4, num_layers=2, bidirectional=True, batch_first=True); bi_output, (bi_h, bi_c) = bi_lstm(torch.randn(2, 5, 3))  # 综合多层双向LSTM。
assert bi_output.shape == (2, 5, 8) and bi_h.shape == bi_c.shape == (4, 2, 4)  # 层和方向合并在状态首维。
