"""
题目 014：Autograd梯度计算_变式

要求：完成“Autograd梯度计算”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建需要梯度的叶子Tensor。
2. 构建标量计算图。
3. 反向传播计算dy/dx。
4. 手算解析梯度。
5. 使用autograd.grad计算一阶和二阶导。

完成标准：
- 验证Autograd结果。
- 验证高阶梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
x = torch.tensor(4.0, requires_grad=True)  # 创建需要梯度的叶子Tensor。
y = x**3 + 2 * x**2 - x  # 构建标量计算图。
y.backward()  # 反向传播计算dy/dx。
expected = 3 * x.detach() ** 2 + 4 * x.detach() - 1  # 手算解析梯度。
assert torch.allclose(x.grad, expected)  # 验证Autograd结果。
print(y.item(), x.grad.item(), y.grad_fn)  # 输出函数值、梯度和计算图节点。
grad_input = torch.tensor(3.0, requires_grad=True); first_grad = torch.autograd.grad(grad_input**3, grad_input, create_graph=True)[0]; second_grad = torch.autograd.grad(first_grad, grad_input)[0]  # 使用autograd.grad计算一阶和二阶导。
assert first_grad.item() == 27 and second_grad.item() == 18  # 验证高阶梯度。
