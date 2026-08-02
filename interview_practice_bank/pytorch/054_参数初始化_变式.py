"""
题目 054：参数初始化_变式

要求：完成“参数初始化”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. Xavier适合tanh或线性激活的方差传播。
4. 验证初始化。

完成标准：
- 验证初始化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
xavier_layer = nn.Linear(8, 4); nn.init.xavier_uniform_(xavier_layer.weight); nn.init.zeros_(xavier_layer.bias)  # Xavier适合tanh或线性激活的方差传播。
assert torch.count_nonzero(xavier_layer.bias) == 0 and torch.isfinite(xavier_layer.weight).all()  # 验证初始化。
