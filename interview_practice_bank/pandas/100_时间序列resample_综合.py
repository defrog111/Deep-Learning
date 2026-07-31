"""
题目 100：时间序列resample_综合

要求：完成“时间序列resample”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建半日频率时间索引。
2. 构造时间序列。
3. 降采样到每日并多重聚合。
4. 移动时间索引而不移动值。

完成标准：
- 验证重采样未丢数据。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
index = pd.date_range('2025-01-01', periods=10, freq='12h')  # 创建半日频率时间索引。
series = pd.Series(range(10), index=index, name='value')  # 构造时间序列。
daily = series.resample('D').agg(['sum', 'mean', 'count'])  # 降采样到每日并多重聚合。
shifted = series.shift(4, freq='h')  # 移动时间索引而不移动值。
assert daily['count'].sum() == len(series)  # 验证重采样未丢数据。
print(daily, '\n', shifted.head())  # 输出每日聚合和索引平移。
