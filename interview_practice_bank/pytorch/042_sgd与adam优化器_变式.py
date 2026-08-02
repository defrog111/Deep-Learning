"""
题目 042：SGD与Adam优化器_变式

要求：完成“SGD与Adam优化器”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 参数组可为不同层设置学习率。
4. 验证分层学习率。

完成标准：
- 验证分层学习率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
group_model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)); group_optimizer = torch.optim.Adam([{'params': group_model[0].parameters(), 'lr': 1e-3}, {'params': group_model[1].parameters(), 'lr': 1e-2}])  # 参数组可为不同层设置学习率。
assert [group['lr'] for group in group_optimizer.param_groups] == [1e-3, 1e-2]  # 验证分层学习率。
