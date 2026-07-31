"""
题目 035：二分类与BCEWithLogitsLoss_易错点

要求：完成“二分类与BCEWithLogitsLoss”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
logits = torch.tensor([1.5, -0.5, 0.2], requires_grad=True)  # 二分类模型输出一个logit。
targets = torch.tensor([1.0, 0.0, 1.0])  # BCE目标必须为浮点0或1。
loss = nn.BCEWithLogitsLoss()(logits, targets)  # 数值稳定地组合sigmoid和BCE。
loss.backward()  # 反向传播。
probabilities = logits.detach().sigmoid()  # 推理阶段用sigmoid得到正类概率。
predictions = probabilities.ge(0.46).int()  # 使用可调阈值得到类别。
assert predictions.shape == targets.shape  # 验证预测shape。
print(loss.item(), probabilities, predictions)  # 输出二分类结果。
