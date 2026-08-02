"""
题目 018：PyTorch nnLinear矩阵等价式

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 创建 nn.Linear 并固定参数。
2. 比较 layer(x) 与 xW 转置+b。
3. 反向传播检查参数梯度 shape。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络层。
layer = nn.Linear(in_features=3, out_features=2)  # 创建保存 weight=(2,3) 的线性层。
with torch.no_grad(): layer.weight.copy_(torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]))  # 固定权重便于核对。
with torch.no_grad(): layer.bias.copy_(torch.tensor([0.5, -0.5]))  # 固定偏置。
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 创建 batch 输入。
official = layer(x)  # nn.Linear 自动计算 x@weight.T+bias。
manual = x @ layer.weight.t() + layer.bias  # 显式写出矩阵等价式。
official.sum().backward()  # 反向传播得到 weight 和 bias 梯度。
assert torch.allclose(official, manual) and layer.weight.grad.shape == (2, 3)  # 核对前向和梯度 shape。
print(official, layer.weight.grad, layer.bias.grad)  # 输出线性层结果。
