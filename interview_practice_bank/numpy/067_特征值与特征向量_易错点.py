"""
题目 067：特征值与特征向量_易错点

要求：完成“特征值与特征向量”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建实对称矩阵。
2. eigh利用对称结构并返回有序实特征值。
3. 重建对称矩阵。

完成标准：
- 重建对称矩阵。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
symmetric = (matrix + matrix.T) / 2; symmetric_values, symmetric_vectors = np.linalg.eigh(symmetric)  # eigh利用对称结构并返回有序实特征值。
assert np.allclose(symmetric, symmetric_vectors @ np.diag(symmetric_values) @ symmetric_vectors.T)  # 重建对称矩阵。
