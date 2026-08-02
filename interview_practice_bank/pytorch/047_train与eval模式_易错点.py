"""
题目 047：train与eval模式_易错点

要求：完成“train与eval模式”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. eval使用并冻结running统计量。
4. 验证推理不更新BatchNorm状态。

完成标准：
- 验证推理不更新BatchNorm状态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
batch_norm = nn.BatchNorm1d(2); batch_norm.train(); batch_norm(torch.tensor([[1.0, 2.0], [3.0, 4.0]])); running_before = batch_norm.running_mean.clone(); batch_norm.eval(); batch_norm(torch.tensor([[100.0, 200.0]]))  # eval使用并冻结running统计量。
assert torch.equal(batch_norm.running_mean, running_before)  # 验证推理不更新BatchNorm状态。
