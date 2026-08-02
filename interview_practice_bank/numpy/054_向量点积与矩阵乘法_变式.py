"""
题目 054：向量点积与矩阵乘法_变式

要求：完成“向量点积与矩阵乘法”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建第一个向量。
2. 创建2×3矩阵。
3. 比较@、dot和matmul的二维行为。

完成标准：
- 验证三种写法等价。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
vector_a = np.array([1.0, 2.0, 3.0])  # 创建第一个向量。
matrix = np.arange(6).reshape(2, 3)  # 创建2×3矩阵。
via_operator = matrix @ vector_a; via_dot = np.dot(matrix, vector_a); via_matmul = np.matmul(matrix, vector_a)  # 比较@、dot和matmul的二维行为。
assert np.allclose(via_operator, via_dot) and np.allclose(via_dot, via_matmul)  # 验证三种写法等价。
