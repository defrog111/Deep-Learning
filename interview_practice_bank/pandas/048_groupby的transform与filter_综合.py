"""
题目 048：GroupBy的transform与filter_综合

要求：完成“GroupBy的transform与filter”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造分组数据。
2. transform返回与原表等长结果。
3. 计算组内中心化分数。
4. filter按组保留原始行。

完成标准：
- 验证各组中心化后和为零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': list('AAABBB'), 'score': [70, 80, 90, 40, 50, 60]})  # 构造分组数据。
frame['team_mean'] = frame.groupby('team')['score'].transform('mean')  # transform返回与原表等长结果。
frame['centered'] = frame['score'] - frame['team_mean']  # 计算组内中心化分数。
large_groups = frame.groupby('team').filter(lambda group: group['score'].mean() >= 56)  # filter按组保留原始行。
assert frame.groupby('team')['centered'].sum().abs().lt(1e-9).all()  # 验证各组中心化后和为零。
print(frame, '\n', large_groups)  # 输出transform和filter结果。
