"""
CSV数据处理练习 016：布尔筛选与query

题目：读取销售CSV，计算净销售额，并用布尔mask和query两种方式筛选高价值电子产品订单。

操作过程：
1. 定位销售CSV。
2. 读取销售数据。
3. 使用eval计算净额。
4. 定义高价值阈值。
5. 定义需要筛选的类别。
6. 使用布尔mask筛选。
7. 使用query并引用两个外部变量。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 读取销售数据。
frame['net'] = frame.eval('quantity * unit_price * (1 - discount)')  # 使用eval计算净额。
minimum = 300  # 定义高价值阈值。
target_category = 'Electronics'  # 定义需要筛选的类别。
by_mask = frame[(frame['category'] == 'Electronics') & frame['net'].ge(minimum)]  # 使用布尔mask筛选。
by_query = frame.query('category == @target_category and net >= @minimum')  # 使用query并引用两个外部变量。
assert by_mask.index.equals(by_query.index)  # 验证两种筛选形式等价。
print(by_query[['order_id', 'product', 'net']])  # 输出高价值订单。
