"""
题目 015：Autograd梯度计算_易错点

要求：完成“Autograd梯度计算”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
x = torch.tensor(5.0, requires_grad=True)  # 创建需要梯度的叶子Tensor。
y = x**3 + 2 * x**2 - x  # 构建标量计算图。
y.backward()  # 反向传播计算dy/dx。
expected = 3 * x.detach() ** 2 + 4 * x.detach() - 1  # 手算解析梯度。
assert torch.allclose(x.grad, expected)  # 验证Autograd结果。
print(y.item(), x.grad.item(), y.grad_fn)  # 输出函数值、梯度和计算图节点。
