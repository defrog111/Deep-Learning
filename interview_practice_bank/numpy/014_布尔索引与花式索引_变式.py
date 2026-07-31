"""
题目 014：布尔索引与花式索引_变式

要求：完成“布尔索引与花式索引”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
array = np.array([3, 8, 1, 9, 4, 7])  # 创建一维数组。
selected = array[array >= 4]  # 使用布尔mask筛选满足阈值的元素。
order = np.array([3, 0, 5])  # 定义花式索引顺序。
reordered = array[order]  # 花式索引总是返回副本。
assert reordered.tolist() == [9, 3, 7]  # 验证按指定位置重排。
print(selected, reordered)  # 输出两种索引结果。
