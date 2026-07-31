"""
题目 006：arange与linspace_变式

要求：完成“arange与linspace”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
integers = np.arange(0, 16, 4)  # 按步长生成左闭右开整数序列。
points = np.linspace(0.0, 1.0, num=6, endpoint=True)  # 按点数生成包含终点的等距序列。
differences = np.diff(points)  # 计算相邻点间隔。
assert np.allclose(differences, differences[0])  # 验证linspace等距。
print(integers, points)  # 输出两种序列。
