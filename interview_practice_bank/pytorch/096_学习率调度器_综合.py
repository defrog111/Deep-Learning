"""
题目 096：学习率调度器_综合

要求：完成“学习率调度器”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建可优化模型。
2. 设置初始学习率。
3. 每指定轮数衰减学习率。
4. 保存学习率轨迹。
5. 模拟六个epoch。
6. 清除梯度。
7. 构造并反向传播简单损失。
8. 更新模型参数。
9. 在epoch末更新学习率。
10. 记录当前学习率。

完成标准：
- 验证学习率未增长。
- 综合验证先升后降轨迹。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
model = nn.Linear(2, 1)  # 创建可优化模型。
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 设置初始学习率。
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)  # 每指定轮数衰减学习率。
history = []  # 保存学习率轨迹。
for _ in range(6):  # 模拟六个epoch。
    optimizer.zero_grad()  # 清除梯度。
    model(torch.ones(1, 2)).sum().backward()  # 构造并反向传播简单损失。
    optimizer.step()  # 更新模型参数。
    scheduler.step()  # 在epoch末更新学习率。
    history.append(scheduler.get_last_lr()[0])  # 记录当前学习率。
assert history[-1] <= history[0]  # 验证学习率未增长。
print(history)  # 输出调度轨迹。
cycle_parameter = nn.Parameter(torch.tensor(1.0)); cycle_optimizer = torch.optim.SGD([cycle_parameter], lr=0.01); one_cycle = torch.optim.lr_scheduler.OneCycleLR(cycle_optimizer, max_lr=0.1, total_steps=5); cycle_lrs = []  # OneCycle按batch而不是epoch更新。
for _ in range(5):  # 模拟五个batch。
    cycle_optimizer.step(); one_cycle.step(); cycle_lrs.append(cycle_optimizer.param_groups[0]['lr'])  # 每个batch更新一次。
assert max(cycle_lrs) > cycle_lrs[-1]  # 综合验证先升后降轨迹。
