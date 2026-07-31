"""
题目 075：GRU序列分类_易错点

要求：完成“GRU序列分类”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建单层GRU。
3. 创建三分类头。
4. 创建五条长度七的序列。
5. 运行序列编码。
6. 使用最终隐藏状态分类。
7. 创建类别标签。
8. 计算分类损失。
9. GRUCell手动控制单时间步循环。
10. 区分GRU模块和GRUCell。

完成标准：
- 验证分类输出shape。
- 区分GRU模块和GRUCell。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入GRU和线性层。
torch.manual_seed(0)  # 固定随机种子。
gru = nn.GRU(input_size=4, hidden_size=8, batch_first=True)  # 创建单层GRU。
head = nn.Linear(8, 3)  # 创建三分类头。
sequence = torch.randn(5, 7, 4)  # 创建五条长度七的序列。
outputs, hidden = gru(sequence)  # 运行序列编码。
logits = head(hidden[-1])  # 使用最终隐藏状态分类。
labels = torch.tensor([0, 1, 2, 1, 0])  # 创建类别标签。
loss = nn.CrossEntropyLoss()(logits, labels)  # 计算分类损失。
loss.backward()  # 验证整个模型可反向传播。
assert logits.shape == (5, 3)  # 验证分类输出shape。
print(loss.item(), outputs.shape, hidden.shape)  # 输出损失和状态shape。
gru_cell = nn.GRUCell(3, 4); cell_hidden = torch.zeros(2, 4); cell_hidden = gru_cell(torch.randn(2, 3), cell_hidden)  # GRUCell手动控制单时间步循环。
assert cell_hidden.shape == (2, 4)  # 区分GRU模块和GRUCell。
