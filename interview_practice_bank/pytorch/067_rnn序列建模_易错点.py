"""
题目 067：RNN序列建模_易错点

要求：完成“RNN序列建模”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 定义序列维度。
3. 创建两层RNN。
4. 创建batch-first输入。
5. 取得所有时间步输出和各层最终状态。
6. 对比顶层最后输出与最终隐藏状态。

完成标准：
- 验证shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
