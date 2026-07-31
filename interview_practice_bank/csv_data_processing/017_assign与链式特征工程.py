"""
CSV数据处理练习 017：assign与链式特征工程

题目：读取销售CSV，使用assign一次创建原价、折扣金额、净额和平均单价特征。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 按声明顺序引用新列。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 读取销售数据。
featured = frame.assign(gross=lambda data: data['quantity'] * data['unit_price'], discount_amount=lambda data: data['gross'] * data['discount'], net=lambda data: data['gross'] - data['discount_amount'], unit_net=lambda data: data['net'] / data['quantity'])  # 按声明顺序引用新列。
assert (featured['net'] <= featured['gross']).all() and featured[['gross', 'discount_amount', 'net', 'unit_net']].notna().all().all()  # 验证金额关系。
print(featured[['order_id', 'gross', 'discount_amount', 'net', 'unit_net']].head())  # 输出衍生特征。
