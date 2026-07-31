"""
题目 077：map替换与apply_基础

要求：完成“map替换与apply”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造映射数据。
2. 一对一字典映射。
3. 对单列执行自定义函数。
4. 替换指定值。

完成标准：
- 验证映射无遗漏。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'grade': ['A', 'B', 'C', 'A'], 'score': [95, 82, 70, 91]})  # 构造映射数据。
frame['points'] = frame['grade'].map({'A': 4, 'B': 3, 'C': 2})  # 一对一字典映射。
frame['bucket'] = frame['score'].apply(lambda value: 'high' if value >= 90 else 'regular')  # 对单列执行自定义函数。
frame['grade'] = frame['grade'].replace({'A': 'Excellent'})  # 替换指定值。
assert frame['points'].notna().all()  # 验证映射无遗漏。
print(frame)  # 输出转换结果。
