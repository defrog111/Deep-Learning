"""
题目 037：unique集合操作与计数_基础

要求：完成“unique集合操作与计数”的基础题，写出关键数组的shape并解释结果。

操作步骤：
1. 构造含重复元素的数组。
2. 同时取得唯一值、首次位置和频次。
3. 定义第一集合。
4. 定义第二集合。
5. 计算交集。

完成标准：
- 验证频次总和等于元素数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([3, 1, 3, 2, 1, 3])  # 构造含重复元素的数组。
unique, first_indices, counts = np.unique(values, return_index=True, return_counts=True)  # 同时取得唯一值、首次位置和频次。
left = np.array([1, 2, 3])  # 定义第一集合。
right = np.array([3, 4, 5])  # 定义第二集合。
intersection = np.intersect1d(left, right)  # 计算交集。
assert counts.sum() == values.size  # 验证频次总和等于元素数。
print(unique, first_indices, counts, intersection)  # 输出集合统计。
