"""
题目 017：NumPy 矩阵范数与余弦相似度

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算向量 L1/L2 范数。
2. 计算矩阵 Frobenius 和谱范数。
3. 批量计算余弦相似度并避免除零。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
vector = np.array([3.0, -4.0])  # 创建二维向量。
matrix = np.array([[3.0, 0.0], [0.0, 4.0]])  # 创建对角矩阵。
l1 = np.linalg.norm(vector, ord=1)  # 计算绝对值和。
l2 = np.linalg.norm(vector, ord=2)  # 计算欧氏范数。
frobenius = np.linalg.norm(matrix, ord='fro')  # 计算所有元素平方和开根号。
spectral = np.linalg.norm(matrix, ord=2)  # 矩阵 2-范数等于最大奇异值。
rows = np.array([[1.0, 0.0], [1.0, 1.0]])  # 创建待比较的行向量。
cosine = (rows @ vector) / (np.linalg.norm(rows, axis=1) * np.linalg.norm(vector) + 1e-12)  # 批量计算余弦相似度。
assert l1 == 7.0 and l2 == frobenius == 5.0 and spectral == 4.0  # 核对不同范数。
print(l1, l2, frobenius, spectral, cosine)  # 输出范数与相似度。
