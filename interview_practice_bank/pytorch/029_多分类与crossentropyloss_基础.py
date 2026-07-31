"""
题目 029：多分类与CrossEntropyLoss_基础

要求：完成“多分类与CrossEntropyLoss”的基础题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], requires_grad=True)  # CrossEntropy接收未归一化logits。
labels = torch.tensor([0, 1])  # 多分类标签使用Long类别索引。
loss = nn.CrossEntropyLoss(label_smoothing=0.02)(logits, labels)  # 内部组合log_softmax和NLLLoss。
loss.backward()  # 计算logits梯度。
probabilities = logits.detach().softmax(dim=1)  # 仅展示时转换概率。
assert torch.allclose(probabilities.sum(1), torch.ones(2))  # 验证每行概率和为1。
print(loss.item(), probabilities, logits.grad)  # 输出损失、概率和梯度。
