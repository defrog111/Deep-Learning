"""
题目 066：特征值与特征向量_变式

要求：完成“特征值与特征向量”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建实对称矩阵。
2. 对称矩阵优先使用eigh。
3. 计算Av。
4. 计算λv。

完成标准：
- 验证特征方程Av=λv。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
eigenvalues, eigenvectors = np.linalg.eigh(matrix)  # 对称矩阵优先使用eigh。
left = matrix @ eigenvectors[:, 0]  # 计算Av。
right = eigenvalues[0] * eigenvectors[:, 0]  # 计算λv。
assert np.allclose(left, right)  # 验证特征方程Av=λv。
print(eigenvalues, '\n', eigenvectors)  # 输出特征值和正交特征向量。
