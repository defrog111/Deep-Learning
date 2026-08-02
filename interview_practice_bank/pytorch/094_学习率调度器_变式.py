"""
题目 094：学习率调度器_变式

要求：完成“学习率调度器”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 创建余弦退火调度器。
4. 模拟四个epoch。
    cosine_optimizer.step(); cosine_scheduler.step(); cosine_history.append(cosine_scheduler.get_last_lr()[0])  # optimizer后调用scheduler。
5. 验证余弦下降。

完成标准：
- 验证余弦下降。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
cosine_parameter = nn.Parameter(torch.tensor(1.0)); cosine_optimizer = torch.optim.SGD([cosine_parameter], lr=0.1); cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cosine_optimizer, T_max=4); cosine_history = []  # 创建余弦退火调度器。
for _ in range(4):  # 模拟四个epoch。
    cosine_optimizer.step(); cosine_scheduler.step(); cosine_history.append(cosine_scheduler.get_last_lr()[0])  # optimizer后调用scheduler。
assert cosine_history[-1] < cosine_history[0]  # 验证余弦下降。
