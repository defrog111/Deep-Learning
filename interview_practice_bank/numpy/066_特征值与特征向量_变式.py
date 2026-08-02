"""
题目 066：特征值与特征向量_变式

要求：完成“特征值与特征向量”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建实对称矩阵。
3. 一般方阵使用eig，实对称矩阵优先eigh。
4. 验证Av等于lambda乘v。

完成标准：
- 验证Av等于lambda乘v。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
general_values, general_vectors = np.linalg.eig(matrix)  # 一般方阵使用eig，实对称矩阵优先eigh。
assert np.allclose(matrix @ general_vectors, general_vectors * general_values)  # 验证Av等于lambda乘v。
