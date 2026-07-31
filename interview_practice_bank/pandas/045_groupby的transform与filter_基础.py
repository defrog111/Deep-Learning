"""
题目 045：GroupBy的transform与filter_基础

要求：完成“GroupBy的transform与filter”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': list('AAABBB'), 'score': [70, 80, 90, 40, 50, 60]})  # 构造分组数据。
frame['team_mean'] = frame.groupby('team')['score'].transform('mean')  # transform返回与原表等长结果。
frame['centered'] = frame['score'] - frame['team_mean']  # 计算组内中心化分数。
large_groups = frame.groupby('team').filter(lambda group: group['score'].mean() >= 53)  # filter按组保留原始行。
assert frame.groupby('team')['centered'].sum().abs().lt(1e-9).all()  # 验证各组中心化后和为零。
print(frame, '\n', large_groups)  # 输出transform和filter结果。
