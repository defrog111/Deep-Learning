"""
CSV数据处理练习 014：按时间插值缺失值

题目：读取传感器CSV并按设备、时间排序，在每台设备内部线性插值温度。

操作过程：
1. 定位传感器CSV。
2. 解析并建立正确时序。
3. 只在设备内部插值。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values(['device_id', 'timestamp'])  # 解析并建立正确时序。
frame['temperature_interpolated'] = frame.groupby('device_id')['temperature'].transform(lambda values: values.interpolate(limit_direction='both'))  # 只在设备内部插值。
assert not frame['temperature_interpolated'].isna().any()  # 验证边界与内部缺失均被处理。
print(frame.loc[frame['temperature'].isna(), ['device_id', 'timestamp', 'temperature_interpolated']])  # 输出插值结果。
