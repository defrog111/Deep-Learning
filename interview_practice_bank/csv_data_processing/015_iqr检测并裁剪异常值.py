"""
CSV数据处理练习 015：IQR检测并裁剪异常值

题目：读取传感器CSV，用IQR规则识别异常温度，并使用clip限制到上下界。

操作过程：
1. 定位传感器CSV。
2. 读取温度数据。
3. 计算四分位数。
4. 计算四分位距。
5. 定义Tukey异常界限。
6. 筛选非缺失异常值。
7. 温莎化裁剪异常值。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sensor_readings.csv'  # 定位传感器CSV。
frame = pd.read_csv(csv_path)  # 读取温度数据。
first_quartile, third_quartile = frame['temperature'].quantile([0.25, 0.75])  # 计算四分位数。
iqr = third_quartile - first_quartile  # 计算四分位距。
lower, upper = first_quartile - 1.5 * iqr, third_quartile + 1.5 * iqr  # 定义Tukey异常界限。
outliers = frame[~frame['temperature'].between(lower, upper) & frame['temperature'].notna()]  # 筛选非缺失异常值。
frame['temperature_clipped'] = frame['temperature'].clip(lower, upper)  # 温莎化裁剪异常值。
assert len(outliers) >= 1 and frame['temperature_clipped'].max() <= upper  # 验证异常识别和裁剪。
print(outliers[['device_id', 'timestamp', 'temperature']], lower, upper)  # 输出异常记录和边界。
