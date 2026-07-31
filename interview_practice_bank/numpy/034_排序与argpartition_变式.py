"""
题目 034：排序与argpartition_变式

要求：完成“排序与argpartition”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
values = np.array([9, 1, 7, 3, 8, 2])  # 构造无序数组。
sorted_values = np.sort(values)  # 完整排序并返回副本。
k = 3  # 设置需要取得的前k个元素数量。
smallest_indices = np.argpartition(values, k - 1)[:k]  # 线性期望时间取得最小k个位置但内部无序。
smallest = np.sort(values[smallest_indices])  # 只对选中的k个元素排序。
assert np.array_equal(smallest, sorted_values[:k])  # 与完整排序结果核对。
print(sorted_values, smallest_indices, smallest)  # 输出排序与局部分区结果。
