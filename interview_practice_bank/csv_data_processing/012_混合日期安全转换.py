"""
CSV数据处理练习 012：混合日期安全转换

题目：读取脏订单CSV，用format=mixed处理不同日期格式，并把非法日期转换为NaT。

操作过程：
1. 定位脏订单CSV。
2. 先按字符串读取日期。
3. 解析混合格式并容忍非法值。
4. 取得无法解析的原始记录。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path)  # 先按字符串读取日期。
frame['parsed_date'] = pd.to_datetime(frame['order_date'], format='mixed', errors='coerce')  # 解析混合格式并容忍非法值。
invalid_rows = frame[frame['parsed_date'].isna()]  # 取得无法解析的原始记录。
assert len(invalid_rows) == 1 and invalid_rows.iloc[0]['order_date'] == 'not-a-date'  # 验证坏日期被隔离。
print(invalid_rows[['order_id', 'order_date']])  # 输出待人工修复记录。
