"""
CSV数据处理练习 004：parse_dates读取日期

题目：读取销售CSV时直接解析日期，再提取年、月、星期和季度。

操作过程：
1. 定位销售CSV。
2. 在读取时把date解析为datetime64。
3. 创建日期特征。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path, parse_dates=['date'])  # 在读取时把date解析为datetime64。
frame = frame.assign(year=frame['date'].dt.year, month=frame['date'].dt.month, weekday=frame['date'].dt.day_name(), quarter=frame['date'].dt.quarter)  # 创建日期特征。
assert frame['date'].dtype.kind == 'M' and frame['month'].between(1, 12).all()  # 验证日期类型和月份范围。
print(frame[['date', 'year', 'month', 'weekday', 'quarter']].head())  # 输出日期特征。
