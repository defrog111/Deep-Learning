"""
题目 054：向量点积与矩阵乘法_变式

要求：完成“向量点积与矩阵乘法”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
vector_a = np.array([1.0, 2.0, 3.0])  # 创建第一个向量。
vector_b = np.array([4.0, 5.0, 6.0])  # 创建第二个向量。
dot = vector_a @ vector_b  # 一维@一维得到标量点积。
matrix = np.arange(6).reshape(2, 3)  # 创建2×3矩阵。
product = matrix @ vector_a  # 矩阵乘向量得到长度2结果。
batch = np.matmul(np.ones((4, 2, 3)), np.ones((4, 3, 5)))  # 批量矩阵乘法。
assert dot == 32 and batch.shape == (4, 2, 5)  # 验证点积值和批量shape。
print(dot, product, batch.shape)  # 输出矩阵运算结果。
