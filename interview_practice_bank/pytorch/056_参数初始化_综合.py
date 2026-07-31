"""
题目 056：参数初始化_综合

要求：完成“参数初始化”的综合题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.Linear(64, 32)  # 创建线性层。
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # 为ReLU网络使用Kaiming初始化变式4。
nn.init.zeros_(layer.bias)  # 偏置初始化为零。
fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)  # 取得扇入扇出用于解释方差。
assert torch.count_nonzero(layer.bias) == 0  # 验证偏置为零。
print(layer.weight.mean().item(), layer.weight.std().item(), fan_in, fan_out)  # 输出初始化统计。
