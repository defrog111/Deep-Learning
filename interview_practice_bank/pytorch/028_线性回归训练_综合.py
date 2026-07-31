"""
题目 028：线性回归训练_综合

要求：完成“线性回归训练”的综合题，说明训练态、梯度和张量shape。

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.09)  # 创建SGD优化器。
for _ in range(150):  # 执行固定轮数训练。
    loss = nn.functional.mse_loss(model(x), y)  # 计算均方误差。
    optimizer.zero_grad()  # 清除上一轮累积梯度。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
assert loss.item() < 0.01  # 验证模型学会线性关系。
print(model.weight.item(), model.bias.item(), loss.item())  # 输出拟合参数和损失。
amp_model = nn.Linear(2, 1); amp_optimizer = torch.optim.SGD(amp_model.parameters(), lr=0.01); amp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp_model.to(amp_device)  # 自动选择可用设备。
with torch.autocast(device_type=amp_device.type, enabled=amp_device.type == 'cuda'):  # CUDA可用时开启自动混合精度。
    amp_loss = amp_model(torch.ones(2, 2, device=amp_device)).pow(2).mean()  # 前向在autocast区域执行。
amp_loss.backward(); amp_optimizer.step(); assert torch.isfinite(amp_loss)  # CPU和GPU路径都能完成训练一步。
