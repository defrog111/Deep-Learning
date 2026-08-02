"""
题目 080：Conv2d输出shape_综合

要求：完成“Conv2d输出shape”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入卷积层。
3. 综合深度可分离卷积。
4. 验证depthwise加pointwise。

完成标准：
- 验证depthwise加pointwise。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入卷积层。
depthwise = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3); pointwise = nn.Conv2d(3, 6, kernel_size=1); separable_output = pointwise(depthwise(torch.randn(2, 3, 8, 8)))  # 综合深度可分离卷积。
assert separable_output.shape == (2, 6, 8, 8)  # 验证depthwise加pointwise。
