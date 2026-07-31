"""
题目 080：Conv2d输出shape_综合

要求：完成“Conv2d输出shape”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 设置输入空间尺寸。
2. 设置卷积核大小。
3. 创建二维卷积。
4. 创建NCHW图像batch。
5. 运行卷积。
6. 根据卷积公式计算输出边长。
7. 综合深度可分离卷积。

完成标准：
- 验证输出shape。
- 验证depthwise加pointwise。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入卷积层。
height = width = 32  # 设置输入空间尺寸。
kernel = 5  # 设置卷积核大小。
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel, stride=2, padding=1)  # 创建二维卷积。
images = torch.randn(4, 3, height, width)  # 创建NCHW图像batch。
features = conv(images)  # 运行卷积。
expected = (height + 2 - kernel) // 2 + 1  # 根据卷积公式计算输出边长。
assert features.shape == (4, 8, expected, expected)  # 验证输出shape。
print(features.shape, expected)  # 输出实际与公式结果。
depthwise = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3); pointwise = nn.Conv2d(3, 6, kernel_size=1); separable_output = pointwise(depthwise(torch.randn(2, 3, 8, 8)))  # 综合深度可分离卷积。
assert separable_output.shape == (2, 6, 8, 8)  # 验证depthwise加pointwise。
