"""
题目 079：Conv2d输出shape_易错点

要求：完成“Conv2d输出shape”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入卷积层。
3. dilation=2使有效卷积核尺寸为5。
4. 验证padding保持空间尺寸。

完成标准：
- 验证padding保持空间尺寸。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入卷积层。
dilated_conv = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2); dilated_output = dilated_conv(torch.randn(1, 1, 10, 10))  # dilation=2使有效卷积核尺寸为5。
assert dilated_output.shape[-2:] == (10, 10)  # 验证padding保持空间尺寸。
