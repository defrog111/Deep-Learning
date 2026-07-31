"""
题目 042：SGD与Adam优化器_变式

要求：完成“SGD与Adam优化器”的变式题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定初始化。
model_sgd = nn.Linear(2, 1)  # 创建SGD模型。
model_adam = nn.Linear(2, 1)  # 创建Adam模型。
model_adam.load_state_dict(model_sgd.state_dict())  # 让两个模型从相同参数开始。
x = torch.tensor([[1.0, 2.0]])  # 创建单个样本。
target = torch.tensor([[1.0]])  # 创建回归目标。
optimizers = [torch.optim.SGD(model_sgd.parameters(), lr=0.04), torch.optim.Adam(model_adam.parameters(), lr=0.01)]  # 配置两种优化器。
for model, optimizer in zip([model_sgd, model_adam], optimizers):  # 分别执行一步更新。
    loss = (model(x) - target).square().mean()  # 计算MSE。
    optimizer.zero_grad()  # 清梯度。
    loss.backward()  # 算梯度。
    optimizer.step()  # 更新参数。
print(model_sgd.weight, model_adam.weight)  # 对比一步后的参数。
