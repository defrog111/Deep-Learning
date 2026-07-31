"""
题目 071：LSTM与隐藏状态_易错点

要求：完成“LSTM与隐藏状态”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建双向两层LSTM。
3. 创建(B,T,F)输入。
4. LSTM返回hidden和cell两类状态。
5. 双向输出特征为2×hidden。
6. 层数×方向数为4。

完成标准：
- 双向输出特征为2×hidden。
- 层数×方向数为4。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入LSTM模块。
torch.manual_seed(0)  # 固定随机种子。
lstm = nn.LSTM(input_size=5, hidden_size=7, num_layers=2, batch_first=True, bidirectional=True)  # 创建双向两层LSTM。
sequence = torch.randn(4, 6, 5)  # 创建(B,T,F)输入。
outputs, (hidden, cell) = lstm(sequence)  # LSTM返回hidden和cell两类状态。
assert outputs.shape == (4, 6, 14)  # 双向输出特征为2×hidden。
assert hidden.shape == cell.shape == (4, 4, 7)  # 层数×方向数为4。
print(outputs.shape, hidden.shape, cell.shape)  # 输出关键shape。
