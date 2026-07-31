"""
CSV数据处理练习 005：读取脏CSV并识别问题

题目：读取脏订单CSV，检查空值、重复订单、状态拼写和金额字符串问题。

操作过程：
1. 定位脏订单CSV。
2. 保留原始脏值用于审计。
3. 汇总常见质量问题。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path)  # 保留原始脏值用于审计。
audit = {'missing': frame.isna().sum().to_dict(), 'duplicate_orders': int(frame.duplicated('order_id').sum()), 'statuses': sorted(frame['status'].dropna().unique())}  # 汇总常见质量问题。
assert audit['duplicate_orders'] == 1 and audit['missing']['amount'] == 1  # 验证识别出重复和缺失金额。
print(audit)  # 输出数据质量审计。
