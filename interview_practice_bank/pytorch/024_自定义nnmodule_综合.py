"""
题目 024：自定义nnModule_综合

要求：完成“自定义nnModule”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入神经网络模块。
3. 注册前向hook并及时移除。
4. 综合验证hook和冻结参数。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
hook_values = []; hook_layer = nn.Linear(2, 1); handle = hook_layer.register_forward_hook(lambda module, inputs, output: hook_values.append(output.shape)); hook_layer(torch.ones(3, 2)); handle.remove()  # 注册前向hook并及时移除。
hook_layer.weight.requires_grad_(False); assert hook_values == [torch.Size([3, 1])] and not hook_layer.weight.requires_grad  # 综合验证hook和冻结参数。
