"""
题目 096：学习率调度器_综合

要求：完成“学习率调度器”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. OneCycle按batch而不是epoch更新。
2. 模拟五个batch。
    cycle_optimizer.step(); one_cycle.step(); cycle_lrs.append(cycle_optimizer.param_groups[0]['lr'])  # 每个batch更新一次。
3. 综合验证先升后降轨迹。

完成标准：
- 综合验证先升后降轨迹。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
cycle_parameter = nn.Parameter(torch.tensor(1.0)); cycle_optimizer = torch.optim.SGD([cycle_parameter], lr=0.01); one_cycle = torch.optim.lr_scheduler.OneCycleLR(cycle_optimizer, max_lr=0.1, total_steps=5); cycle_lrs = []  # OneCycle按batch而不是epoch更新。
for _ in range(5):  # 模拟五个batch。
    cycle_optimizer.step(); one_cycle.step(); cycle_lrs.append(cycle_optimizer.param_groups[0]['lr'])  # 每个batch更新一次。
assert max(cycle_lrs) > cycle_lrs[-1]  # 综合验证先升后降轨迹。
