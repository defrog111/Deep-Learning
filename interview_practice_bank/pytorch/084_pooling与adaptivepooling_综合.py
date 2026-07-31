"""
题目 084：Pooling与AdaptivePooling_综合

要求：完成“Pooling与AdaptivePooling”的综合题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入池化模块。
features = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)  # 创建NCHW特征图。
max_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(features)  # 最大池化把空间尺寸减半。
adaptive = nn.AdaptiveAvgPool2d((4, 4))(features)  # 自适应池化直接指定输出尺寸。
global_average = nn.AdaptiveAvgPool2d(1)(features).flatten(1)  # 全局平均池化得到每通道特征。
assert max_pooled.shape == (2, 3, 4, 4) and global_average.shape == (2, 3)  # 验证shape。
print(max_pooled.shape, adaptive.shape, global_average)  # 输出池化结果。
