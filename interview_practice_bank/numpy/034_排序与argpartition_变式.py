"""
题目 034：排序与argpartition_变式

要求：完成“排序与argpartition”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 对结构化数组多键排序。
3. 验证多字段排序。

完成标准：
- 验证多字段排序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
records = np.array([(2, 'b'), (1, 'c'), (1, 'a')], dtype=[('score', int), ('name', 'U1')]); ordered_records = np.sort(records, order=['score', 'name'])  # 对结构化数组多键排序。
assert ordered_records['name'].tolist() == ['a', 'c', 'b']  # 验证多字段排序。
