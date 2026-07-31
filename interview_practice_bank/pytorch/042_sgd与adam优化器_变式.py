"""
题目 042：SGD与Adam优化器_变式

要求：完成“SGD与Adam优化器”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定初始化。
2. 创建SGD模型。
3. 创建Adam模型。
4. 让两个模型从相同参数开始。
5. 创建单个样本。
6. 创建回归目标。
7. 配置两种优化器。
8. 分别执行一步更新。
9. 计算MSE。
10. 清梯度。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
