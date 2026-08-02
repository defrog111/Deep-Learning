"""
题目 038：unique集合操作与计数_变式

要求：完成“unique集合操作与计数”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 构造含重复元素的数组。
2. 一次取得唯一值、首次位置、逆映射和计数。
3. 用逆映射重建原数组。

完成标准：
- 用逆映射重建原数组。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([3, 1, 3, 2, 1, 3])  # 构造含重复元素的数组。
unique_values, first_indices, inverse, counts = np.unique(values, return_index=True, return_inverse=True, return_counts=True)  # 一次取得唯一值、首次位置、逆映射和计数。
assert np.array_equal(unique_values[inverse], values) and counts.sum() == values.size  # 用逆映射重建原数组。
