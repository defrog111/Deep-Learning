"""
题目 044：where与select条件选择_综合

要求：完成“where与select条件选择”的综合题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
values = np.array([-3, 0, 2, 8, 12])  # 构造跨多个条件区间的数据。
sign = np.where(values >= 0, 1, -1)  # 二选一条件向量化。
labels = np.select([values < 0, values < 10], ['negative', 'small'], default='large')  # 多条件按顺序选择。
clipped = np.clip(values, 0, 10)  # 把数值限制到闭区间。
assert labels.tolist() == ['negative', 'small', 'small', 'small', 'large']  # 验证条件优先级。
print(sign, labels, clipped)  # 输出条件运算结果。
