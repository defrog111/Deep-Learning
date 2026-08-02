"""
题目 004：PyTorch einsum常用矩阵操作

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用 einsum 完成转置、迹和矩阵乘法。
2. 解释重复下标代表收缩。
3. 与专用 API 核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
matrix = torch.arange(1.0, 10.0).reshape(3, 3)  # 创建 3×3 方阵。
transposed = torch.einsum('ij->ji', matrix)  # 交换下标顺序实现转置。
trace = torch.einsum('ii->', matrix)  # 对重复的对角下标求和。
product = torch.einsum('ik,kj->ij', matrix, matrix)  # 收缩 k 实现矩阵乘法。
assert torch.equal(transposed, matrix.t()) and trace == torch.trace(matrix)  # 核对转置和迹。
assert torch.equal(product, matrix @ matrix)  # 核对 einsum 矩阵乘法。
print(transposed, trace, product)  # 输出 einsum 结果。
