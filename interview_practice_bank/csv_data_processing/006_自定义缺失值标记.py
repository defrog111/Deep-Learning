"""
CSV数据处理练习 006：自定义缺失值标记

题目：读取传感器CSV时把missing状态和空字符串作为缺失标记，并统计每台设备缺失量。

操作过程：
1. 定位传感器CSV。
2. 扩展默认缺失值标记。
3. 按设备统计传感器缺失。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, na_values=['', 'missing'], keep_default_na=True)  # 扩展默认缺失值标记。
missing_by_device = frame.groupby('device_id')[['temperature', 'humidity']].apply(lambda group: group.isna().sum())  # 按设备统计传感器缺失。
assert frame[['temperature', 'humidity']].isna().any().any()  # 验证数值缺失被识别。
print(missing_by_device)  # 输出设备缺失统计。
