"""
题目 066：RNN序列建模_变式

要求：完成“RNN序列建模”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入RNN模块。
3. 双向RNN拼接两个方向特征。
4. 验证方向维。

完成标准：
- 验证方向维。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入RNN模块。
bidirectional_rnn = nn.RNN(3, 4, batch_first=True, bidirectional=True); bidirectional_output, bidirectional_hidden = bidirectional_rnn(torch.randn(2, 5, 3))  # 双向RNN拼接两个方向特征。
assert bidirectional_output.shape == (2, 5, 8) and bidirectional_hidden.shape == (2, 2, 4)  # 验证方向维。
