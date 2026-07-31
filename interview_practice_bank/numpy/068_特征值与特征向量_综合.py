"""
题目 068：特征值与特征向量_综合

要求：完成“特征值与特征向量”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建实对称矩阵。
2. 对称矩阵优先使用eigh。
3. 计算Av。
4. 计算λv。
5. 用一次幂迭代逼近主特征向量方向。

完成标准：
- 验证特征方程Av=λv。
- 验证归一化。
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
power_vector = np.ones(matrix.shape[0]); power_vector = matrix @ power_vector; power_vector /= np.linalg.norm(power_vector)  # 用一次幂迭代逼近主特征向量方向。
assert np.isclose(np.linalg.norm(power_vector), 1)  # 验证归一化。
