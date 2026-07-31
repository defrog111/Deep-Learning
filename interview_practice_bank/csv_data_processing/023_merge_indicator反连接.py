"""
CSV数据处理练习 023：merge indicator反连接

题目：读取客户和订单CSV，找出没有任何订单的客户以及找不到客户的订单。

操作过程：
1. 定位共享数据目录。
2. 读取客户主表。
3. 读取唯一订单。
4. 定义反连接的indicator标记。
5. 从客户侧检查订单匹配。
6. 取得没有订单的客户。
7. 从订单侧检查客户匹配。
8. 取得孤儿订单。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
data_folder = Path(__file__).parents[1] / 'data'  # 定位共享数据目录。
customers = pd.read_csv(data_folder / 'customers.csv')  # 读取客户主表。
orders = pd.read_csv(data_folder / 'orders_dirty.csv').drop_duplicates('order_id')  # 读取唯一订单。
left_only = 'left_only'  # 定义反连接的indicator标记。
customer_audit = customers.merge(orders[['customer_id']].drop_duplicates(), on='customer_id', how='left', indicator=True)  # 从客户侧检查订单匹配。
customers_without_orders = customer_audit.query('_merge == @left_only')  # 取得没有订单的客户。
order_audit = orders.merge(customers[['customer_id']], on='customer_id', how='left', indicator=True)  # 从订单侧检查客户匹配。
unknown_customer_orders = order_audit.query('_merge == @left_only')  # 取得孤儿订单。
assert len(customers_without_orders) == 0 and unknown_customer_orders['customer_id'].tolist() == ['C999']  # 验证两个反连接结果。
print(customers_without_orders, unknown_customer_orders[['order_id', 'customer_id']], sep='\n')  # 输出数据完整性问题。
