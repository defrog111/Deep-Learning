"""
题目 036：二分类与BCEWithLogitsLoss_综合

要求：完成“二分类与BCEWithLogitsLoss”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. 多标签分类每类独立使用BCE。
4. 区分多标签BCE和互斥多类CE。

完成标准：
- 区分多标签BCE和互斥多类CE。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
multi_logits = torch.tensor([[2.0, -1.0, 0.0]]); multi_targets = torch.tensor([[1.0, 0.0, 1.0]]); multi_loss = nn.BCEWithLogitsLoss(reduction='none')(multi_logits, multi_targets)  # 多标签分类每类独立使用BCE。
assert multi_loss.shape == multi_targets.shape and torch.isfinite(multi_loss).all()  # 区分多标签BCE和互斥多类CE。
