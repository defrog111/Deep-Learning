"""
题目 095：学习率调度器_易错点

要求：完成“学习率调度器”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. Plateau调度器step接收验证指标而不是epoch。
4. 验证无改进时降学习率。

完成标准：
- 验证无改进时降学习率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
plateau_parameter = nn.Parameter(torch.tensor(1.0)); plateau_optimizer = torch.optim.SGD([plateau_parameter], lr=0.1); plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(plateau_optimizer, mode='min', patience=0, factor=0.5); [plateau_scheduler.step(metric) for metric in [1.0, 1.1, 1.2]]  # Plateau调度器step接收验证指标而不是epoch。
assert plateau_optimizer.param_groups[0]['lr'] < 0.1  # 验证无改进时降学习率。
