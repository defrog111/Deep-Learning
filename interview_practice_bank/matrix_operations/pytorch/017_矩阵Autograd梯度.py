"""
题目 017：PyTorch 矩阵Autograd梯度

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 构建 y=XW+b 和 MSE。
2. 使用 backward 取得矩阵梯度。
3. 手算 W 的梯度并核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 创建固定输入矩阵 X。
weight = torch.tensor([[0.5], [1.0]], requires_grad=True)  # 创建需要梯度的权重 W。
target = torch.tensor([[3.0], [7.0]])  # 创建回归目标。
prediction = x @ weight  # 计算矩阵形式线性预测。
residual = prediction - target  # 保存残差用于手算梯度。
loss = residual.square().mean()  # 计算所有元素 MSE。
loss.backward()  # Autograd 计算 dLoss/dW。
manual_gradient = (2.0 / target.numel()) * x.t() @ residual.detach()  # 根据链式法则手算矩阵梯度。
assert torch.allclose(weight.grad, manual_gradient)  # 核对自动和手工梯度。
print(loss.item(), weight.grad, manual_gradient)  # 输出梯度检查结果。
