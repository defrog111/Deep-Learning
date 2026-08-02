"""
题目 075：GRU序列分类_易错点

要求：完成“GRU序列分类”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入GRU和线性层。
3. GRUCell手动控制单时间步循环。
4. 区分GRU模块和GRUCell。

完成标准：
- 区分GRU模块和GRUCell。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入GRU和线性层。
gru_cell = nn.GRUCell(3, 4); cell_hidden = torch.zeros(2, 4); cell_hidden = gru_cell(torch.randn(2, 3), cell_hidden)  # GRUCell手动控制单时间步循环。
assert cell_hidden.shape == (2, 4)  # 区分GRU模块和GRUCell。
