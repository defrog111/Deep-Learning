"""
题目 082：Pooling与AdaptivePooling_变式

要求：完成“Pooling与AdaptivePooling”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建NCHW特征图。
2. 最大池化把空间尺寸减半。
3. 自适应池化直接指定输出尺寸。
4. 全局平均池化得到每通道特征。

完成标准：
- 验证shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入池化模块。
features = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)  # 创建NCHW特征图。
max_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(features)  # 最大池化把空间尺寸减半。
adaptive = nn.AdaptiveAvgPool2d((2, 2))(features)  # 自适应池化直接指定输出尺寸。
global_average = nn.AdaptiveAvgPool2d(1)(features).flatten(1)  # 全局平均池化得到每通道特征。
assert max_pooled.shape == (2, 3, 4, 4) and global_average.shape == (2, 3)  # 验证shape。
print(max_pooled.shape, adaptive.shape, global_average)  # 输出池化结果。
