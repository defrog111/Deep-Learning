"""
CSV数据处理练习 020：GroupBy transform计算占比

题目：读取销售CSV，计算每笔订单净额占所在地区总净额的比例。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 计算订单净额。
4. 把地区总额广播回原始行。
5. 计算组内占比。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 读取销售数据。
frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算订单净额。
frame['region_total'] = frame.groupby('region')['net'].transform('sum')  # 把地区总额广播回原始行。
frame['region_share'] = frame['net'] / frame['region_total']  # 计算组内占比。
assert frame.groupby('region')['region_share'].sum().round(10).eq(1).all()  # 验证各地区占比和为1。
print(frame[['order_id', 'region', 'net', 'region_share']].head())  # 输出组内占比。
