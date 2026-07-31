"""
CSV数据处理练习 031：时间序列resample重采样

题目：读取传感器CSV，把每台设备的小时数据重采样为每日均值和有效观测数。

操作过程：
1. 定位传感器CSV。
2. 使用时间索引。
3. 显式选择数值列后按设备和天重采样。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, parse_dates=['timestamp']).set_index('timestamp')  # 使用时间索引。
daily = frame.groupby('device_id')[['temperature', 'humidity']].resample('D').agg(temperature_mean=('temperature', 'mean'), humidity_mean=('humidity', 'mean'), valid_temperature=('temperature', 'count')).reset_index()  # 显式选择数值列后按设备和天重采样。
assert len(daily) == frame['device_id'].nunique() * 2 and daily['valid_temperature'].gt(0).all()  # 验证每设备每天一行。
print(daily)  # 输出每日汇总。
