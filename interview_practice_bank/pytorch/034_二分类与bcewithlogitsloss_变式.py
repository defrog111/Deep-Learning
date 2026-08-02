"""
题目 034：二分类与BCEWithLogitsLoss_变式

要求：完成“二分类与BCEWithLogitsLoss”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. pos_weight只放大正类项。
4. 验证正类加权。

完成标准：
- 验证正类加权。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
imbalanced_logits = torch.tensor([0.0, 0.0]); imbalanced_targets = torch.tensor([0.0, 1.0]); positive_weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0), reduction='none')(imbalanced_logits, imbalanced_targets)  # pos_weight只放大正类项。
assert positive_weighted[1] == positive_weighted[0] * 4  # 验证正类加权。
