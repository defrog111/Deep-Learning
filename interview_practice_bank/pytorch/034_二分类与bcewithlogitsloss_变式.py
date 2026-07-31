"""
题目 034：二分类与BCEWithLogitsLoss_变式

要求：完成“二分类与BCEWithLogitsLoss”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 二分类模型输出一个logit。
2. BCE目标必须为浮点0或1。
3. 数值稳定地组合sigmoid和BCE。
4. 反向传播。
5. 推理阶段用sigmoid得到正类概率。
6. 使用可调阈值得到类别。

完成标准：
- 验证预测shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
logits = torch.tensor([1.5, -0.5, 0.2], requires_grad=True)  # 二分类模型输出一个logit。
targets = torch.tensor([1.0, 0.0, 1.0])  # BCE目标必须为浮点0或1。
loss = nn.BCEWithLogitsLoss()(logits, targets)  # 数值稳定地组合sigmoid和BCE。
loss.backward()  # 反向传播。
probabilities = logits.detach().sigmoid()  # 推理阶段用sigmoid得到正类概率。
predictions = probabilities.ge(0.44).int()  # 使用可调阈值得到类别。
assert predictions.shape == targets.shape  # 验证预测shape。
print(loss.item(), probabilities, predictions)  # 输出二分类结果。
