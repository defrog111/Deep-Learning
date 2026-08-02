"""
题目 001：PyTorch dot_mv_mm基础

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 分别使用 dot、mv 和 mm。
2. 写出每个输入和输出 shape。
3. 与 @ 运算符核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
vector = torch.tensor([1.0, 2.0, 3.0])  # 创建一维向量。
matrix = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])  # 创建二维矩阵。
dot = torch.dot(vector, vector)  # dot 只接受两个一维 Tensor。
matrix_vector = torch.mv(matrix, vector)  # mv 专门执行二维矩阵乘一维向量。
matrix_matrix = torch.mm(matrix, matrix.t())  # mm 专门执行两个二维矩阵相乘。
assert dot.item() == 14.0 and torch.equal(matrix_vector, matrix @ vector)  # 与 @ 核对向量结果。
assert torch.equal(matrix_matrix, matrix @ matrix.t())  # 与 @ 核对矩阵结果。
print(dot, matrix_vector, matrix_matrix)  # 输出三类运算。
