"""
题目 030：多分类与CrossEntropyLoss_变式

要求：完成“多分类与CrossEntropyLoss”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. 类别权重放大少数类损失。
4. 两样本置信度相同但第二类权重更大。

完成标准：
- 两样本置信度相同但第二类权重更大。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
weighted_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]]); weighted_labels = torch.tensor([0, 1]); weighted_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]), reduction='none')(weighted_logits, weighted_labels)  # 类别权重放大少数类损失。
assert weighted_loss[1] > weighted_loss[0]  # 两样本置信度相同但第二类权重更大。
