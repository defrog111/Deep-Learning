"""
题目 078：outer与einsum_变式

要求：完成“outer与einsum”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1, 2, 3])  # 创建列因子向量。
b = np.array([4, 5])  # 创建行因子向量。
outer = np.outer(a, b)  # 计算所有两两乘积。
einsum_outer = np.einsum('i,j->ij', a, b)  # 用爱因斯坦求和表达外积。
matrix = np.arange(6).reshape(2, 3)  # 创建矩阵。
column_sums = np.einsum('ij->j', matrix)  # 用einsum沿行轴求和。
assert np.array_equal(outer, einsum_outer)  # 验证两种外积写法一致。
print(outer, column_sums)  # 输出einsum结果。
