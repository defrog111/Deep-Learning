"""
题目 009：切片与负索引_基础

要求：完成“切片与负索引”的基础题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
center = matrix[1:3, 1:3]  # 同时切片行和列取得中心块。
reversed_rows = matrix[::-1]  # 使用负步长翻转行顺序。
every_n = matrix.ravel()[::1]  # 在展平视图上按步长抽样。
assert center.shape == (2, 2)  # 验证二维切片shape。
print(center, '\n', reversed_rows, '\n', every_n)  # 输出切片结果。
