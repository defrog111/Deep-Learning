"""
题目 001：NumPy matmul维度规则

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 分别完成向量点积、矩阵乘向量、矩阵乘矩阵。
2. 写出每种输入和输出 shape。
3. 用断言验证手算结果。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
vector = np.array([1.0, 2.0, 3.0])  # 创建 shape=(3,) 的向量。
matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])  # 创建 shape=(2,3) 的矩阵。
dot = vector @ vector  # (3,)@(3,) 收缩唯一轴并得到标量。
matrix_vector = matrix @ vector  # (2,3)@(3,) 得到 shape=(2,)。
matrix_matrix = matrix @ matrix.T  # (2,3)@(3,2) 得到 shape=(2,2)。
assert dot == 14.0 and matrix_vector.shape == (2,) and matrix_matrix.shape == (2, 2)  # 核对值与 shape。
print(dot, matrix_vector, matrix_matrix)  # 输出三类乘法结果。
