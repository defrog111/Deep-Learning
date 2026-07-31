"""
题目 011：loc与iloc选择_易错点

要求：完成“loc与iloc选择”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建非默认索引。
2. loc 使用索引标签选择。
3. iloc 使用整数位置选择。
4. 组合行条件和列选择。
5. 证明本例标签20恰好位于位置1。

完成标准：
- 证明本例标签20恰好位于位置1。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'name': ['A', 'B', 'C'], 'score': [78, 92, 85]}, index=[10, 20, 30])  # 创建非默认索引。
by_label = frame.loc[20, 'score']  # loc 使用索引标签选择。
by_position = frame.iloc[1, 1]  # iloc 使用整数位置选择。
subset = frame.loc[frame['score'].gt(80), ['name', 'score']]  # 组合行条件和列选择。
assert by_label == by_position == 92  # 证明本例标签20恰好位于位置1。
print(subset)  # 输出筛选结果。
