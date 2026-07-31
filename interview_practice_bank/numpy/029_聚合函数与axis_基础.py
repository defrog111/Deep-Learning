"""
题目 029：聚合函数与axis_基础

要求：完成“聚合函数与axis”的基础题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
cube = np.arange(24).reshape(2, 3, 4)  # 创建三维数组。
sum_axis0 = cube.sum(axis=0)  # 消去batch轴得到shape(3,4)。
mean_last = cube.mean(axis=-1, keepdims=True)  # 消去最后轴但保留长度1维。
maximum_positions = cube.argmax(axis=2)  # 返回每行最大值的位置。
assert mean_last.shape == (2, 3, 1)  # 验证keepdims便于后续广播。
print(sum_axis0, '\n', mean_last, '\n', maximum_positions)  # 输出聚合结果。
