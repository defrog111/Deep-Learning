"""
CSV数据处理练习 022：merge客户与订单

题目：读取客户CSV和脏订单CSV，清洗订单金额后执行many-to-one左连接。

操作过程：
1. 定位共享数据目录。
2. 读取客户主表。
3. 读取并去重订单。
4. 清洗金额。
5. 验证多订单对一客户关系。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
data_folder = Path(__file__).parents[1] / 'data'  # 定位共享数据目录。
customers = pd.read_csv(data_folder / 'customers.csv')  # 读取客户主表。
orders = pd.read_csv(data_folder / 'orders_dirty.csv').drop_duplicates('order_id')  # 读取并去重订单。
orders['amount'] = pd.to_numeric(orders['amount'].astype('string').str.replace(r'[$,]', '', regex=True), errors='coerce')  # 清洗金额。
merged = orders.merge(customers, on='customer_id', how='left', validate='many_to_one', indicator=True)  # 验证多订单对一客户关系。
assert len(merged) == len(orders) and (merged['_merge'] == 'left_only').sum() == 1  # 验证左连接保行并发现未知客户。
print(merged[['order_id', 'customer_id', 'customer_name', 'amount', '_merge']])  # 输出连接结果。
