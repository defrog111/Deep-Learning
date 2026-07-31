"""
题目 067：RNN序列建模_易错点

要求：完成“RNN序列建模”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入RNN模块。
torch.manual_seed(0)  # 固定随机种子。
batch, steps, features, hidden = 3, 5, 4, 6  # 定义序列维度。
rnn = nn.RNN(features, hidden, num_layers=2, batch_first=True, nonlinearity='tanh')  # 创建两层RNN。
sequence = torch.randn(batch, steps, features)  # 创建batch-first输入。
outputs, final_hidden = rnn(sequence)  # 取得所有时间步输出和各层最终状态。
assert outputs.shape == (batch, steps, hidden) and final_hidden.shape == (2, batch, hidden)  # 验证shape。
print(outputs[:, -1], final_hidden[-1])  # 对比顶层最后输出与最终隐藏状态。
