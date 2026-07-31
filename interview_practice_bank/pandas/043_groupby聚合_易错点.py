"""
题目 043：GroupBy聚合_易错点

要求：完成“GroupBy聚合”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 读取销售数据。
2. 计算净销售额。
3. 命名聚合。
4. 按总销售额降序。

完成标准：
- 验证分组订单数守恒。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入 Pandas。
frame = pd.read_csv(Path(__file__).parents[1] / 'data' / 'sales.csv')  # 读取销售数据。
frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算净销售额。
summary = frame.groupby('region', as_index=False).agg(total_net=('net', 'sum'), orders=('order_id', 'nunique'), avg_quantity=('quantity', 'mean'))  # 命名聚合。
summary = summary.sort_values('total_net', ascending=False)  # 按总销售额降序。
assert summary['orders'].sum() == frame['order_id'].nunique()  # 验证分组订单数守恒。
print(summary)  # 输出地区汇总。
