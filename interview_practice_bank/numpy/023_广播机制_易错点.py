"""
题目 023：广播机制_易错点

要求：完成“广播机制”的易错点题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(12).reshape(3, 4)  # 创建shape为(3,4)的矩阵。
column_bias = np.array([10, 20, 30, 40])  # 创建shape为(4,)的列方向偏置。
row_scale = np.array([1, 2, 3]).reshape(3, 1)  # 显式变成shape为(3,1)的行方向缩放。
answer = (matrix + column_bias) * row_scale  # 利用广播完成无循环计算。
assert answer.shape == matrix.shape  # 广播结果保持矩阵shape。
print(answer)  # 输出广播运算结果。
