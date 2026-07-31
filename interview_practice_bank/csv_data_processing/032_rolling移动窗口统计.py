"""
CSV数据处理练习 032：rolling移动窗口统计

题目：读取传感器CSV，按设备计算3期移动平均和移动标准差。

操作过程：
1. 定位传感器CSV。
2. 建立设备内时间顺序。
3. 计算3期移动均值。
4. 至少两点才计算标准差。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values(['device_id', 'timestamp'])  # 建立设备内时间顺序。
frame['temperature_ma3'] = frame.groupby('device_id')['temperature'].transform(lambda values: values.rolling(3, min_periods=1).mean())  # 计算3期移动均值。
frame['temperature_std3'] = frame.groupby('device_id')['temperature'].transform(lambda values: values.rolling(3, min_periods=2).std())  # 至少两点才计算标准差。
assert frame.groupby('device_id')['temperature_ma3'].first().notna().all()  # 验证每台设备首行有均值。
print(frame[['timestamp', 'device_id', 'temperature', 'temperature_ma3', 'temperature_std3']].head(8))  # 输出窗口特征。
