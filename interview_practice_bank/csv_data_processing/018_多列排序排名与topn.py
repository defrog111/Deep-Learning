"""
CSV数据处理练习 018：多列排序排名与TopN

题目：读取销售CSV，计算净额、地区内排名，并取得每个地区最高订单。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 计算排序指标。
4. 计算地区内无跳号排名。
5. 每区保留第一名并指定并列规则。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 读取销售数据。
frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算排序指标。
frame['region_rank'] = frame.groupby('region')['net'].rank(method='dense', ascending=False).astype(int)  # 计算地区内无跳号排名。
top_each_region = frame.sort_values(['region', 'net', 'order_id'], ascending=[True, False, True]).groupby('region', as_index=False).head(1)  # 每区保留第一名并指定并列规则。
assert top_each_region['region'].is_unique and top_each_region['region_rank'].eq(1).all()  # 验证每区一条第一名。
print(top_each_region[['region', 'order_id', 'net', 'region_rank']])  # 输出地区Top1。
