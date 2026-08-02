"""
题目 058：梯度裁剪_变式

要求：完成“梯度裁剪”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. clip_grad_value_逐元素截断梯度。
4. 区分按值裁剪和全局范数裁剪。

完成标准：
- 区分按值裁剪和全局范数裁剪。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
clip_parameter = nn.Parameter(torch.tensor([10.0, -10.0])); clip_parameter.grad = torch.tensor([5.0, -5.0]); nn.utils.clip_grad_value_([clip_parameter], clip_value=1.0)  # clip_grad_value_逐元素截断梯度。
assert clip_parameter.grad.tolist() == [1.0, -1.0]  # 区分按值裁剪和全局范数裁剪。
