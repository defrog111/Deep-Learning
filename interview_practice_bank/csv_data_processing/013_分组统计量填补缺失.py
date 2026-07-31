"""
CSV数据处理练习 013：分组统计量填补缺失

题目：读取传感器CSV，使用每台设备自己的温度中位数填补缺失值。

操作过程：
1. 定位传感器CSV。
2. 读取传感器数据。
3. 返回与原表等长的设备中位数。
4. 按行使用对应设备统计量填补。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path)  # 读取传感器数据。
device_median = frame.groupby('device_id')['temperature'].transform('median')  # 返回与原表等长的设备中位数。
frame['temperature_filled'] = frame['temperature'].fillna(device_median)  # 按行使用对应设备统计量填补。
assert not frame['temperature_filled'].isna().any()  # 验证温度缺失全部解决。
print(frame.loc[frame['temperature'].isna(), ['device_id', 'temperature', 'temperature_filled']])  # 查看被填补的行。
