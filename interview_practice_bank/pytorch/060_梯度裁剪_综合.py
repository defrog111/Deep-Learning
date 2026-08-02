"""
题目 060：梯度裁剪_综合

要求：完成“梯度裁剪”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 对全部参数统一计算全局范数。
4. 验证综合裁剪结果。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
parameters_for_clip = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))]; [setattr(parameter, 'grad', torch.full_like(parameter, 3.0)) for parameter in parameters_for_clip]; original_norm = nn.utils.clip_grad_norm_(parameters_for_clip, max_norm=1.0)  # 对全部参数统一计算全局范数。
new_norm = torch.sqrt(sum(parameter.grad.pow(2).sum() for parameter in parameters_for_clip)); assert original_norm > 1 and new_norm <= 1.00001  # 验证综合裁剪结果。
