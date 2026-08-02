"""
题目 044：SGD与Adam优化器_综合

要求：完成“SGD与Adam优化器”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 综合执行优化和高效清梯度。
4. 验证参数更新与grad状态。

完成标准：
- 验证参数更新与grad状态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
compare_parameter = nn.Parameter(torch.tensor(1.0)); sgd = torch.optim.SGD([compare_parameter], lr=0.1, momentum=0.9); compare_parameter.grad = torch.tensor(2.0); sgd.step(); sgd.zero_grad(set_to_none=True)  # 综合执行优化和高效清梯度。
assert compare_parameter.item() < 1 and compare_parameter.grad is None  # 验证参数更新与grad状态。
