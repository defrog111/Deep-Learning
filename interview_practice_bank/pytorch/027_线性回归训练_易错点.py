"""
题目 027：线性回归训练_易错点

要求：完成“线性回归训练”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入神经网络模块。
3. set_to_none节省写零操作并便于判断未参与反传参数。
4. 验证梯度被设为None。

完成标准：
- 验证梯度被设为None。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
zero_model = nn.Linear(1, 1); zero_model(torch.ones(1, 1)).sum().backward(); zero_model.zero_grad(set_to_none=True)  # set_to_none节省写零操作并便于判断未参与反传参数。
assert all(parameter.grad is None for parameter in zero_model.parameters())  # 验证梯度被设为None。
