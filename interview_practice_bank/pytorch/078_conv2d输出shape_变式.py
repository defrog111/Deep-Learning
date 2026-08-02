"""
题目 078：Conv2d输出shape_变式

要求：完成“Conv2d输出shape”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入卷积层。
3. groups=2把输入输出通道分成两组卷积。
4. 验证分组卷积权重shape。

完成标准：
- 验证分组卷积权重shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入卷积层。
group_conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2); group_output = group_conv(torch.randn(2, 4, 16, 16))  # groups=2把输入输出通道分成两组卷积。
assert group_output.shape == (2, 8, 16, 16) and group_conv.weight.shape == (8, 2, 3, 3)  # 验证分组卷积权重shape。
