"""
CSV数据处理练习 026：pivot_table透视汇总

题目：读取销售CSV，以地区为行、类别为列计算净销售额，并添加总计。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 计算净额。
4. 创建交叉汇总及总计。

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
pivot = frame.pivot_table(index='region', columns='category', values='net', aggfunc='sum', fill_value=0, margins=True, margins_name='Total')  # 创建交叉汇总及总计。
assert 'Total' in pivot.index and 'Total' in pivot.columns  # 验证行列总计存在。
print(pivot)  # 输出销售透视表。
