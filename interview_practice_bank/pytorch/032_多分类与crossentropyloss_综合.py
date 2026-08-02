"""
题目 032：多分类与CrossEntropyLoss_综合

要求：完成“多分类与CrossEntropyLoss”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. 使用ignore_index忽略padding token。
4. 验证序列分类损失。

完成标准：
- 验证序列分类损失。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
sequence_logits = torch.randn(2, 4, 3); sequence_labels = torch.tensor([[0, 1, -100, -100], [2, 1, 0, -100]]); token_loss = nn.CrossEntropyLoss(ignore_index=-100)(sequence_logits.reshape(-1, 3), sequence_labels.reshape(-1))  # 使用ignore_index忽略padding token。
assert token_loss.ndim == 0 and torch.isfinite(token_loss)  # 验证序列分类损失。
