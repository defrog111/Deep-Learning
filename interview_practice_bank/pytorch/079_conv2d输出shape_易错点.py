"""
题目 079：Conv2d输出shape_易错点

要求：完成“Conv2d输出shape”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
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
