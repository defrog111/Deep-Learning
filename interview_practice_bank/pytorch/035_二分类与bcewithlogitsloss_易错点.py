"""
题目 035：二分类与BCEWithLogitsLoss_易错点

要求：完成“二分类与BCEWithLogitsLoss”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入损失函数模块。
3. 0.5概率阈值等价于0 logit阈值。
4. 避免推理时不必要的sigmoid。

完成标准：
- 避免推理时不必要的sigmoid。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
binary_logits = torch.tensor([-1.0, 0.0, 1.0]); by_logit = binary_logits >= 0; by_probability = torch.sigmoid(binary_logits) >= 0.5  # 0.5概率阈值等价于0 logit阈值。
assert torch.equal(by_logit, by_probability)  # 避免推理时不必要的sigmoid。
