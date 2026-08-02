"""
题目 036：排序与argpartition_综合

要求：完成“排序与argpartition”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 构造无序数组。
3. 只确定第k位置时无需完整排序。
4. 验证partition关键位置。

完成标准：
- 验证partition关键位置。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([9, 1, 7, 3, 8, 2])  # 构造无序数组。
median_position = len(values) // 2; partitioned = np.partition(values, median_position)  # 只确定第k位置时无需完整排序。
assert partitioned[median_position] == np.sort(values)[median_position]  # 验证partition关键位置。
