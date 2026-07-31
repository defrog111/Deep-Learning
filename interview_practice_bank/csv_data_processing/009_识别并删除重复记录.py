"""
CSV数据处理练习 009：识别并删除重复记录

题目：读取脏订单CSV，分别找出重复订单的所有行，并按订单号保留最后一条。

操作过程：
1. 定位脏订单CSV。
2. 读取订单。
3. 标记重复键的全部记录。
4. 每个订单保留最后一条。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path)  # 读取订单。
duplicate_rows = frame[frame.duplicated('order_id', keep=False)]  # 标记重复键的全部记录。
deduplicated = frame.drop_duplicates('order_id', keep='last')  # 每个订单保留最后一条。
assert len(duplicate_rows) == 2 and deduplicated['order_id'].is_unique  # 验证重复识别和唯一结果。
print(duplicate_rows, deduplicated.tail(), sep='\n')  # 对比重复行和去重结果。
