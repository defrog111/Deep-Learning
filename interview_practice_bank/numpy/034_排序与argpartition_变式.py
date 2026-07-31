"""
题目 034：排序与argpartition_变式

要求：完成“排序与argpartition”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 构造无序数组。
2. 完整排序并返回副本。
3. 设置需要取得的前k个元素数量。
4. 线性期望时间取得最小k个位置但内部无序。
5. 只对选中的k个元素排序。
6. 与完整排序结果核对。
7. 对结构化数组多键排序。

完成标准：
- 与完整排序结果核对。
- 验证多字段排序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([9, 1, 7, 3, 8, 2])  # 构造无序数组。
sorted_values = np.sort(values)  # 完整排序并返回副本。
k = 3  # 设置需要取得的前k个元素数量。
smallest_indices = np.argpartition(values, k - 1)[:k]  # 线性期望时间取得最小k个位置但内部无序。
smallest = np.sort(values[smallest_indices])  # 只对选中的k个元素排序。
assert np.array_equal(smallest, sorted_values[:k])  # 与完整排序结果核对。
print(sorted_values, smallest_indices, smallest)  # 输出排序与局部分区结果。
records = np.array([(2, 'b'), (1, 'c'), (1, 'a')], dtype=[('score', int), ('name', 'U1')]); ordered_records = np.sort(records, order=['score', 'name'])  # 对结构化数组多键排序。
assert ordered_records['name'].tolist() == ['a', 'c', 'b']  # 验证多字段排序。
