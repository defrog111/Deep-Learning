"""
题目 028：线性回归训练_综合

要求：完成“线性回归训练”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 自动选择可用设备。
2. CUDA可用时开启自动混合精度。
    amp_loss = amp_model(torch.ones(2, 2, device=amp_device)).pow(2).mean()  # 前向在autocast区域执行。
3. CPU和GPU路径都能完成训练一步。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
amp_model = nn.Linear(2, 1); amp_optimizer = torch.optim.SGD(amp_model.parameters(), lr=0.01); amp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp_model.to(amp_device)  # 自动选择可用设备。
with torch.autocast(device_type=amp_device.type, enabled=amp_device.type == 'cuda'):  # CUDA可用时开启自动混合精度。
    amp_loss = amp_model(torch.ones(2, 2, device=amp_device)).pow(2).mean()  # 前向在autocast区域执行。
amp_loss.backward(); amp_optimizer.step(); assert torch.isfinite(amp_loss)  # CPU和GPU路径都能完成训练一步。
