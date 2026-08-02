"""
题目 074：GRU序列分类_变式

要求：完成“GRU序列分类”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入GRU和线性层。
3. GRU只有h没有LSTM的c。
4. 验证双向GRU。

完成标准：
- 验证双向GRU。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入GRU和线性层。
bi_gru = nn.GRU(3, 4, batch_first=True, bidirectional=True); bi_gru_output, bi_gru_hidden = bi_gru(torch.randn(2, 5, 3))  # GRU只有h没有LSTM的c。
assert bi_gru_output.shape == (2, 5, 8) and bi_gru_hidden.shape == (2, 2, 4)  # 验证双向GRU。
