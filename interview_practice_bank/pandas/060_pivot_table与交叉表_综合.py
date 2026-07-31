"""
题目 060：pivot_table与交叉表_综合

要求：完成“pivot_table与交叉表”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造长表。
2. 创建带总计的透视表。
3. 计算组内比例交叉表。

完成标准：
- 验证每组比例和为1。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': ['A', 'A', 'B', 'B'], 'quarter': ['Q1', 'Q2', 'Q1', 'Q2'], 'sales': [10, 15, 20, 18]})  # 构造长表。
pivot = frame.pivot_table(index='team', columns='quarter', values='sales', aggfunc='sum', fill_value=0, margins=True)  # 创建带总计的透视表。
cross = pd.crosstab(frame['team'], frame['quarter'], normalize='index')  # 计算组内比例交叉表。
assert cross.sum(axis=1).round(8).eq(1).all()  # 验证每组比例和为1。
print(pivot, '\n', cross)  # 输出透视表与交叉表。
