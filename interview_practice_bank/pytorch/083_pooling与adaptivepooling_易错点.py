"""
题目 083：Pooling与AdaptivePooling_易错点

要求：完成“Pooling与AdaptivePooling”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入池化模块。
3. ceil_mode决定是否保留不完整末尾窗口。
4. 验证输出尺寸陷阱。

完成标准：
- 验证输出尺寸陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入池化模块。
ceil_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)(torch.ones(1, 1, 4, 4)); floor_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)(torch.ones(1, 1, 4, 4))  # ceil_mode决定是否保留不完整末尾窗口。
assert ceil_pool.shape[-1] == 2 and floor_pool.shape[-1] == 1  # 验证输出尺寸陷阱。
