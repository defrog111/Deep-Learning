"""
CSV数据处理练习 019：GroupBy命名聚合

题目：读取销售CSV，按地区计算净额总和、订单数、平均订单和最大折扣。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 计算净额。
4. 使用命名聚合生成平坦列名。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 读取销售数据。
frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算净额。
summary = frame.groupby('region', as_index=False).agg(total_net=('net', 'sum'), order_count=('order_id', 'nunique'), average_net=('net', 'mean'), max_discount=('discount', 'max'))  # 使用命名聚合生成平坦列名。
assert summary['order_count'].sum() == frame['order_id'].nunique()  # 验证订单数量守恒。
print(summary.sort_values('total_net', ascending=False))  # 输出地区汇总。
