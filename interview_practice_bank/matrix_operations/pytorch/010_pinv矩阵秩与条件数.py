"""
题目 010：PyTorch pinv矩阵秩与条件数

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 构造秩亏矩阵。
2. 计算 matrix_rank、pinv 和 cond。
3. 验证伪逆的 Moore-Penrose 性质。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # 创建秩为1的矩阵。
rank = torch.linalg.matrix_rank(matrix)  # 计算数值秩。
pseudo_inverse = torch.linalg.pinv(matrix)  # 用 SVD 计算 Moore-Penrose 伪逆。
condition = torch.linalg.cond(matrix)  # 秩亏矩阵的条件数非常大或为无穷。
projected = matrix @ pseudo_inverse @ matrix  # 计算 A A+ A。
assert rank.item() == 1 and torch.allclose(projected, matrix, atol=1e-5)  # 验证秩和伪逆性质。
print(rank, pseudo_inverse, condition)  # 输出数值诊断。
