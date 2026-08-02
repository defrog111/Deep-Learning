"""
题目 013：NumPy eig与eigh选择

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 对实对称矩阵优先调用 eigh。
2. 验证 Av 等于 lambda v。
3. 验证特征向量正交并重建矩阵。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
values, vectors = np.linalg.eigh(matrix)  # eigh 利用对称结构并返回升序实特征值。
left = matrix @ vectors  # 同时计算所有 Av。
right = vectors * values  # 每一列特征向量乘对应特征值。
reconstructed = vectors @ np.diag(values) @ vectors.T  # 用 QΛQ 转置重建对称矩阵。
assert np.allclose(left, right) and np.allclose(vectors.T @ vectors, np.eye(2))  # 验证特征方程和正交性。
assert np.allclose(reconstructed, matrix)  # 验证谱分解重建。
print(values, vectors, reconstructed)  # 输出谱分解结果。
