"""
题目 060：pivot_table与交叉表_综合

要求：完成“pivot_table与交叉表”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': ['A', 'A', 'B', 'B'], 'quarter': ['Q1', 'Q2', 'Q1', 'Q2'], 'sales': [10, 15, 20, 18]})  # 构造长表。
pivot = frame.pivot_table(index='team', columns='quarter', values='sales', aggfunc='sum', fill_value=0, margins=True)  # 创建带总计的透视表。
cross = pd.crosstab(frame['team'], frame['quarter'], normalize='index')  # 计算组内比例交叉表。
assert cross.sum(axis=1).round(8).eq(1).all()  # 验证每组比例和为1。
print(pivot, '\n', cross)  # 输出透视表与交叉表。
