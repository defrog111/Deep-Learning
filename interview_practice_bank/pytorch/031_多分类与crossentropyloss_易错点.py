"""
题目 031：多分类与CrossEntropyLoss_易错点

要求：完成“多分类与CrossEntropyLoss”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. CrossEntropyLoss要求原始logits，提前softmax会改变梯度。
4. 验证常见双重softmax错误。

完成标准：
- 验证常见双重softmax错误。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
raw_logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True); correct_ce = nn.CrossEntropyLoss()(raw_logits, torch.tensor([2])); wrong_ce = nn.CrossEntropyLoss()(torch.softmax(raw_logits, dim=1), torch.tensor([2]))  # CrossEntropyLoss要求原始logits，提前softmax会改变梯度。
assert not torch.allclose(correct_ce, wrong_ce)  # 验证常见双重softmax错误。
