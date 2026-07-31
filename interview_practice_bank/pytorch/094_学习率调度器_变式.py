"""
题目 094：学习率调度器_变式

要求：完成“学习率调度器”的变式题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
model = nn.Linear(2, 1)  # 创建可优化模型。
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 设置初始学习率。
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 每指定轮数衰减学习率。
history = []  # 保存学习率轨迹。
for _ in range(6):  # 模拟六个epoch。
    optimizer.zero_grad()  # 清除梯度。
    model(torch.ones(1, 2)).sum().backward()  # 构造并反向传播简单损失。
    optimizer.step()  # 更新模型参数。
    scheduler.step()  # 在epoch末更新学习率。
    history.append(scheduler.get_last_lr()[0])  # 记录当前学习率。
assert history[-1] <= history[0]  # 验证学习率未增长。
print(history)  # 输出调度轨迹。
