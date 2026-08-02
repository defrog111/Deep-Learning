"""
题目 026：线性回归训练_变式

要求：完成“线性回归训练”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 初始化梯度累积训练。
2. 遍历两个微批次。
    (accumulation_model(micro_batch).pow(2).mean() / 2).backward()  # loss除以累积步数保持梯度尺度。
3. 累积完成后只更新一次。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
accumulation_model = nn.Linear(1, 1); accumulation_optimizer = torch.optim.SGD(accumulation_model.parameters(), lr=0.01); accumulation_optimizer.zero_grad(set_to_none=True)  # 初始化梯度累积训练。
for micro_batch in [torch.ones(2, 1), torch.full((2, 1), 2.0)]:  # 遍历两个微批次。
    (accumulation_model(micro_batch).pow(2).mean() / 2).backward()  # loss除以累积步数保持梯度尺度。
accumulation_optimizer.step(); assert accumulation_model.weight.grad is not None  # 累积完成后只更新一次。
