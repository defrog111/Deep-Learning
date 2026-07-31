"""
题目 026：线性回归训练_变式

要求：完成“线性回归训练”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建单特征训练样本。
3. 根据真实线性关系生成标签。
4. 定义线性回归模型。
5. 创建SGD优化器。
6. 执行固定轮数训练。
7. 计算均方误差。
8. 清除上一轮累积梯度。
9. 反向传播。
10. 更新参数。

完成标准：
- 验证模型学会线性关系。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
torch.manual_seed(0)  # 固定随机种子。
x = torch.linspace(-1, 1, 40).unsqueeze(1)  # 创建单特征训练样本。
y = 3 * x + 2  # 根据真实线性关系生成标签。
model = nn.Linear(1, 1)  # 定义线性回归模型。
optimizer = torch.optim.SGD(model.parameters(), lr=0.07)  # 创建SGD优化器。
for _ in range(150):  # 执行固定轮数训练。
    loss = nn.functional.mse_loss(model(x), y)  # 计算均方误差。
    optimizer.zero_grad()  # 清除上一轮累积梯度。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
assert loss.item() < 0.01  # 验证模型学会线性关系。
print(model.weight.item(), model.bias.item(), loss.item())  # 输出拟合参数和损失。
accumulation_model = nn.Linear(1, 1); accumulation_optimizer = torch.optim.SGD(accumulation_model.parameters(), lr=0.01); accumulation_optimizer.zero_grad(set_to_none=True)  # 初始化梯度累积训练。
for micro_batch in [torch.ones(2, 1), torch.full((2, 1), 2.0)]:  # 遍历两个微批次。
    (accumulation_model(micro_batch).pow(2).mean() / 2).backward()  # loss除以累积步数保持梯度尺度。
accumulation_optimizer.step(); assert accumulation_model.weight.grad is not None  # 累积完成后只更新一次。
