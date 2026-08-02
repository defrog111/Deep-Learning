"""
题目 046：train与eval模式_变式

要求：完成“train与eval模式”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. Dropout仅在train随机置零并缩放。
4. 验证训练推理差异。

完成标准：
- 验证训练推理差异。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
dropout = nn.Dropout(p=0.5); dropout.train(); torch.manual_seed(1); train_output = dropout(torch.ones(100)); dropout.eval(); eval_output = dropout(torch.ones(100))  # Dropout仅在train随机置零并缩放。
assert torch.count_nonzero(train_output) < 100 and torch.equal(eval_output, torch.ones(100))  # 验证训练推理差异。
