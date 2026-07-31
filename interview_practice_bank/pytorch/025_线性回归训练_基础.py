"""
题目 025：线性回归训练_基础

要求：完成“线性回归训练”的基础题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
torch.manual_seed(0)  # 固定随机种子。
x = torch.linspace(-1, 1, 40).unsqueeze(1)  # 创建单特征训练样本。
y = 3 * x + 2  # 根据真实线性关系生成标签。
model = nn.Linear(1, 1)  # 定义线性回归模型。
optimizer = torch.optim.SGD(model.parameters(), lr=0.060000000000000005)  # 创建SGD优化器。
for _ in range(150):  # 执行固定轮数训练。
    loss = nn.functional.mse_loss(model(x), y)  # 计算均方误差。
    optimizer.zero_grad()  # 清除上一轮累积梯度。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
assert loss.item() < 0.01  # 验证模型学会线性关系。
print(model.weight.item(), model.bias.item(), loss.item())  # 输出拟合参数和损失。
