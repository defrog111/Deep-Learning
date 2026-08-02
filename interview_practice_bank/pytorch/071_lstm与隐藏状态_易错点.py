"""
题目 071：LSTM与隐藏状态_易错点

要求：完成“LSTM与隐藏状态”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入LSTM模块。
3. 投影LSTM的输出和h为proj_size，c仍为hidden_size。
4. 验证投影shape陷阱。

完成标准：
- 验证投影shape陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入LSTM模块。
projection_lstm = nn.LSTM(3, 6, proj_size=4, batch_first=True); projection_output, (projection_h, projection_c) = projection_lstm(torch.randn(2, 5, 3))  # 投影LSTM的输出和h为proj_size，c仍为hidden_size。
assert projection_output.shape[-1] == 4 and projection_h.shape[-1] == 4 and projection_c.shape[-1] == 6  # 验证投影shape陷阱。
