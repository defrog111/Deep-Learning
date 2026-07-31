"""
题目 079：Conv2d输出shape_易错点

要求：完成“Conv2d输出shape”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 设置输入空间尺寸。
2. 设置卷积核大小。
3. 创建二维卷积。
4. 创建NCHW图像batch。
5. 运行卷积。
6. 根据卷积公式计算输出边长。
7. dilation=2使有效卷积核尺寸为5。

完成标准：
- 验证输出shape。
- 验证padding保持空间尺寸。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入卷积层。
height = width = 32  # 设置输入空间尺寸。
kernel = 3  # 设置卷积核大小。
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel, stride=2, padding=1)  # 创建二维卷积。
images = torch.randn(4, 3, height, width)  # 创建NCHW图像batch。
features = conv(images)  # 运行卷积。
expected = (height + 2 - kernel) // 2 + 1  # 根据卷积公式计算输出边长。
assert features.shape == (4, 8, expected, expected)  # 验证输出shape。
print(features.shape, expected)  # 输出实际与公式结果。
dilated_conv = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2); dilated_output = dilated_conv(torch.randn(1, 1, 10, 10))  # dilation=2使有效卷积核尺寸为5。
assert dilated_output.shape[-2:] == (10, 10)  # 验证padding保持空间尺寸。
