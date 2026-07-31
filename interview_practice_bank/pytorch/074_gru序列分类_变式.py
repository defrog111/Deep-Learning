"""
题目 074：GRU序列分类_变式

要求：完成“GRU序列分类”的变式题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
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
