"""
题目 054：参数初始化_变式

要求：完成“参数初始化”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建线性层。
3. 为ReLU网络使用Kaiming初始化变式2。
4. 偏置初始化为零。
5. 取得扇入扇出用于解释方差。
6. Xavier适合tanh或线性激活的方差传播。

完成标准：
- 验证偏置为零。
- 验证初始化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.Linear(64, 32)  # 创建线性层。
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # 为ReLU网络使用Kaiming初始化变式2。
nn.init.zeros_(layer.bias)  # 偏置初始化为零。
fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)  # 取得扇入扇出用于解释方差。
assert torch.count_nonzero(layer.bias) == 0  # 验证偏置为零。
print(layer.weight.mean().item(), layer.weight.std().item(), fan_in, fan_out)  # 输出初始化统计。
xavier_layer = nn.Linear(8, 4); nn.init.xavier_uniform_(xavier_layer.weight); nn.init.zeros_(xavier_layer.bias)  # Xavier适合tanh或线性激活的方差传播。
assert torch.count_nonzero(xavier_layer.bias) == 0 and torch.isfinite(xavier_layer.weight).all()  # 验证初始化。
