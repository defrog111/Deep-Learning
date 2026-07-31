"""
题目 068：特征值与特征向量_综合

要求：完成“特征值与特征向量”的综合题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
eigenvalues, eigenvectors = np.linalg.eigh(matrix)  # 对称矩阵优先使用eigh。
left = matrix @ eigenvectors[:, 0]  # 计算Av。
right = eigenvalues[0] * eigenvectors[:, 0]  # 计算λv。
assert np.allclose(left, right)  # 验证特征方程Av=λv。
print(eigenvalues, '\n', eigenvectors)  # 输出特征值和正交特征向量。
