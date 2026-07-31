"""
题目 060：梯度裁剪_综合

要求：完成“梯度裁剪”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建简单模型。
2. 创建会产生较大梯度的输入。
3. 构造损失。
4. 计算原始梯度。
5. 按总范数裁剪并返回裁剪前范数。
6. 计算裁剪后总范数。

完成标准：
- 验证梯度范数不超过阈值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
model = nn.Linear(3, 1)  # 创建简单模型。
x = torch.tensor([[100.0, -100.0, 50.0]])  # 创建会产生较大梯度的输入。
loss = model(x).square().mean()  # 构造损失。
loss.backward()  # 计算原始梯度。
before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 按总范数裁剪并返回裁剪前范数。
after = torch.sqrt(sum(parameter.grad.square().sum() for parameter in model.parameters()))  # 计算裁剪后总范数。
assert after <= 1.00001  # 验证梯度范数不超过阈值。
print(before.item(), after.item())  # 输出裁剪前后范数。
