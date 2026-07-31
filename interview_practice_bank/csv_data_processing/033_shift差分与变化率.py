"""
CSV数据处理练习 033：shift差分与变化率

题目：读取传感器CSV，按设备计算上一时刻温度、温度差和百分比变化。

操作过程：
1. 定位传感器CSV。
2. 建立设备内时序。
3. 缓存分组列。
4. 取得同设备上一条温度。
5. 计算一阶差分。
6. 计算变化率且不自动填缺失。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values(['device_id', 'timestamp'])  # 建立设备内时序。
grouped_temperature = frame.groupby('device_id')['temperature']  # 缓存分组列。
frame['previous_temperature'] = grouped_temperature.shift(1)  # 取得同设备上一条温度。
frame['temperature_difference'] = grouped_temperature.diff()  # 计算一阶差分。
frame['temperature_pct_change'] = grouped_temperature.pct_change(fill_method=None)  # 计算变化率且不自动填缺失。
assert frame.groupby('device_id').head(1)['previous_temperature'].isna().all()  # 使用head保留NaN并验证每台设备首行没有历史值。
print(frame[['device_id', 'timestamp', 'temperature', 'previous_temperature', 'temperature_difference', 'temperature_pct_change']].head(8))  # 输出滞后特征。
