"""
题目 073：shift差分与增长率_基础

要求：完成“shift差分与增长率”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造月度序列。
2. 取得上一期数值。
3. 计算绝对差分。
4. 计算环比增长率。

完成标准：
- 验证第二期差额。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'month': pd.date_range('2025-01-01', periods=5, freq='MS'), 'sales': [100, 120, 90, 135, 150]})  # 构造月度序列。
frame['previous'] = frame['sales'].shift(1)  # 取得上一期数值。
frame['change'] = frame['sales'].diff()  # 计算绝对差分。
frame['growth'] = frame['sales'].pct_change()  # 计算环比增长率。
assert frame.loc[1, 'change'] == 20  # 验证第二期差额。
print(frame)  # 输出滞后与增长特征。
