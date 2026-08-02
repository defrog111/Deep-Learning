"""
题目 084：Pooling与AdaptivePooling_综合

要求：完成“Pooling与AdaptivePooling”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入池化模块。
3. 综合全局平均和固定shape最大池化。
4. 验证自适应池化。

完成标准：
- 验证自适应池化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入池化模块。
mixed_pool_input = torch.randn(2, 3, 7, 9); global_average = nn.AdaptiveAvgPool2d(1)(mixed_pool_input).flatten(1); fixed_max = nn.AdaptiveMaxPool2d((2, 3))(mixed_pool_input)  # 综合全局平均和固定shape最大池化。
assert global_average.shape == (2, 3) and fixed_max.shape == (2, 3, 2, 3)  # 验证自适应池化。
