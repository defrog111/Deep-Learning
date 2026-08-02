"""
题目 082：Pooling与AdaptivePooling_变式

要求：完成“Pooling与AdaptivePooling”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入池化模块。
3. 最大池化可返回位置用于unpool。
4. 验证池化值与索引。

完成标准：
- 验证池化值与索引。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入池化模块。
pool_input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4); max_values, max_indices = nn.MaxPool2d(2, return_indices=True)(pool_input)  # 最大池化可返回位置用于unpool。
assert max_values.tolist() == [[[[5, 7], [13, 15]]]] and max_indices.shape == max_values.shape  # 验证池化值与索引。
