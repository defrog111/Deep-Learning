"""
题目 068：RNN序列建模_综合

要求：完成“RNN序列建模”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入RNN模块。
3. dropout只作用于多层RNN的层间连接。
4. 综合验证层数与shape。

完成标准：
- 综合验证层数与shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入RNN模块。
stacked_rnn = nn.RNN(3, 4, num_layers=2, dropout=0.2, batch_first=True); stacked_output, stacked_hidden = stacked_rnn(torch.randn(2, 5, 3))  # dropout只作用于多层RNN的层间连接。
assert stacked_hidden.shape == (2, 2, 4) and stacked_output.shape == (2, 5, 4)  # 综合验证层数与shape。
