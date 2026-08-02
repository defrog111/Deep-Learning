"""
题目 014：Autograd梯度计算_变式

要求：完成“Autograd梯度计算”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 使用autograd.grad计算一阶和二阶导。
3. 验证高阶梯度。

完成标准：
- 验证高阶梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
grad_input = torch.tensor(3.0, requires_grad=True); first_grad = torch.autograd.grad(grad_input**3, grad_input, create_graph=True)[0]; second_grad = torch.autograd.grad(first_grad, grad_input)[0]  # 使用autograd.grad计算一阶和二阶导。
assert first_grad.item() == 27 and second_grad.item() == 18  # 验证高阶梯度。
