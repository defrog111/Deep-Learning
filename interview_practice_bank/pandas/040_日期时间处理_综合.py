"""
题目 040：日期时间处理_综合

要求：完成“日期时间处理”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'timestamp': ['2025-01-01 08:30', '2025-02-15 17:45', 'bad']})  # 构造日期字符串。
frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')  # 非法日期转NaT。
frame['month'] = frame['timestamp'].dt.to_period('M')  # 提取月份周期。
frame['weekday'] = frame['timestamp'].dt.day_name()  # 提取星期名称。
frame['plus_days'] = frame['timestamp'] + pd.Timedelta(days=4)  # 日期加减使用Timedelta。
assert frame['timestamp'].isna().sum() == 1  # 验证非法日期被识别。
print(frame)  # 输出日期衍生特征。
