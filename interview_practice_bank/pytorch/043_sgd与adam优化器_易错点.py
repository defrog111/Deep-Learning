"""
题目 043：SGD与Adam优化器_易错点

要求：完成“SGD与Adam优化器”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. Adam更新后保存动量状态。
4. 只保存模型权重不足以无缝恢复训练。

完成标准：
- 只保存模型权重不足以无缝恢复训练。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
state_model = nn.Linear(1, 1); state_optimizer = torch.optim.Adam(state_model.parameters()); state_model(torch.ones(1, 1)).sum().backward(); state_optimizer.step(); optimizer_state = state_optimizer.state_dict()  # Adam更新后保存动量状态。
assert optimizer_state['state'] and 'param_groups' in optimizer_state  # 只保存模型权重不足以无缝恢复训练。
