"""
CSV数据处理练习 021：GroupBy filter筛选整个组

题目：读取销售CSV，只保留订单总净额超过1500的地区的全部原始订单。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 计算净额。
4. filter按组判断但返回原始行。

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
large_regions = frame.groupby('region').filter(lambda group: group['net'].sum() > 1500)  # filter按组判断但返回原始行。
assert large_regions.groupby('region')['net'].sum().gt(1500).all()  # 验证保留组满足条件。
print(large_regions[['order_id', 'region', 'net']])  # 输出被保留地区的订单。
