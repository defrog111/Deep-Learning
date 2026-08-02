"""
题目 006：NumPy einsum常用单矩阵操作

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用 einsum 求转置、迹和行和。
2. 分别写出下标含义。
3. 与 NumPy 专用函数核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1.0, 10.0).reshape(3, 3)  # 创建 3×3 方阵。
transposed = np.einsum('ij->ji', matrix)  # 交换输出下标顺序实现转置。
trace = np.einsum('ii->', matrix)  # 重复下标 i 表示取对角线并求和。
row_sums = np.einsum('ij->i', matrix)  # 消去 j 下标得到每一行的和。
assert np.array_equal(transposed, matrix.T)  # 与标准转置核对。
assert trace == np.trace(matrix) and np.array_equal(row_sums, matrix.sum(axis=1))  # 与专用函数核对。
print(transposed, trace, row_sums)  # 输出 einsum 结果。
