"""
题目 028：concatenate与split_综合

要求：完成“concatenate与split”的综合题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
left = np.arange(6).reshape(2, 3)  # 创建左侧矩阵。
right = np.arange(6, 12).reshape(2, 3)  # 创建右侧矩阵。
rows = np.concatenate([left, right], axis=0)  # 沿行轴拼接。
columns = np.concatenate([left, right], axis=1)  # 沿列轴拼接。
parts = np.array_split(columns, 3, axis=1)  # 允许不等分地拆成三块。
assert rows.shape == (4, 3) and columns.shape == (2, 6)  # 验证不同轴的shape。
print(rows, '\n', [part.shape for part in parts])  # 输出拼接与拆分结果。
