"""
题目 004：NumPy 转置换轴与共轭转置

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 区分一维向量转置和列向量。
2. 对三维 batch 只交换最后两轴。
3. 对复数矩阵执行共轭转置。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
vector = np.array([1.0, 2.0, 3.0])  # 一维向量没有行列之分。
column = vector[:, None]  # 用新增轴把 (3,) 变成列向量 (3,1)。
batch = np.arange(24).reshape(2, 3, 4)  # 创建三维批量矩阵。
batch_transposed = batch.swapaxes(-1, -2)  # 只交换矩阵的 M、K 两轴得到 (2,4,3)。
complex_matrix = np.array([[1 + 2j, 3 - 1j]])  # 创建复数行矩阵。
adjoint = complex_matrix.conj().T  # 共轭后转置得到 Hermitian transpose。
assert vector.T.shape == (3,) and column.shape == (3, 1)  # 一维 .T 不会变成列向量。
assert batch_transposed.shape == (2, 4, 3) and adjoint.shape == (2, 1)  # 验证换轴结果。
print(column, batch_transposed.shape, adjoint)  # 输出关键结果。
