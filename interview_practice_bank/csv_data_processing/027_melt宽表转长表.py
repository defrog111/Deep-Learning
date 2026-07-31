"""
CSV数据处理练习 027：melt宽表转长表

题目：读取传感器CSV，把温度和湿度两列转换为metric/value长表。

操作过程：
1. 定位传感器CSV。
2. 读取宽表。
3. 把两个测量列堆叠成长表。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path, parse_dates=['timestamp'])  # 读取宽表。
long_frame = frame.melt(id_vars=['timestamp', 'device_id', 'status'], value_vars=['temperature', 'humidity'], var_name='metric', value_name='value')  # 把两个测量列堆叠成长表。
assert len(long_frame) == len(frame) * 2 and set(long_frame['metric']) == {'temperature', 'humidity'}  # 验证行数扩展和指标类别。
print(long_frame.head())  # 输出长表。
