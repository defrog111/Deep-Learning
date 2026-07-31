"""
题目 041：GroupBy聚合_基础

要求：完成“GroupBy聚合”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入 Pandas。
frame = pd.read_csv(Path(__file__).parents[1] / 'data' / 'sales.csv')  # 读取销售数据。
frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算净销售额。
summary = frame.groupby('region', as_index=False).agg(total_net=('net', 'sum'), orders=('order_id', 'nunique'), avg_quantity=('quantity', 'mean'))  # 命名聚合。
summary = summary.sort_values('total_net', ascending=False)  # 按总销售额降序。
assert summary['orders'].sum() == frame['order_id'].nunique()  # 验证分组订单数守恒。
print(summary)  # 输出地区汇总。
