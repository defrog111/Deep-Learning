"""
题目 067：RNN序列建模_易错点

要求：完成“RNN序列建模”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入RNN模块。
3. batch_first=False输入顺序为序列、批次、特征。
4. 防止混淆batch和sequence轴。

完成标准：
- 防止混淆batch和sequence轴。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入RNN模块。
time_major_rnn = nn.RNN(3, 4, batch_first=False); time_output, _ = time_major_rnn(torch.randn(5, 2, 3))  # batch_first=False输入顺序为序列、批次、特征。
assert time_output.shape == (5, 2, 4)  # 防止混淆batch和sequence轴。
