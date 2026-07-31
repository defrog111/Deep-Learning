"""
CSV数据处理练习 042：端到端CSV数据处理流水线

题目：读取客户、订单和评论三个CSV，完成清洗、连接、聚合和客户特征表构建。

操作过程：
1. 定位共享数据目录。
2. 读取客户主表。
3. 读取并去重订单。
4. 读取评论。
5. 金额转数值。
6. 状态标准化。
7. 定义已完成订单状态。
8. 只使用已完成订单构建消费特征。
9. 聚合订单特征。
10. 聚合评论特征。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
data_folder = Path(__file__).parents[1] / 'data'  # 定位共享数据目录。
customers = pd.read_csv(data_folder / 'customers.csv', parse_dates=['signup_date'])  # 读取客户主表。
orders = pd.read_csv(data_folder / 'orders_dirty.csv').drop_duplicates('order_id').copy()  # 读取并去重订单。
reviews = pd.read_csv(data_folder / 'reviews.csv', parse_dates=['created_at'])  # 读取评论。
orders['amount'] = pd.to_numeric(orders['amount'].astype('string').str.replace(r'[$,]', '', regex=True), errors='coerce')  # 金额转数值。
orders['status'] = orders['status'].str.strip().str.lower()  # 状态标准化。
completed_status = 'completed'  # 定义已完成订单状态。
completed = orders.query('status == @completed_status')  # 只使用已完成订单构建消费特征。
order_features = completed.groupby('customer_id').agg(completed_orders=('order_id', 'nunique'), total_spend=('amount', 'sum'), average_order=('amount', 'mean'), last_order=('order_date', 'max'))  # 聚合订单特征。
review_features = reviews.groupby('customer_id').agg(review_count=('review_id', 'nunique'), average_rating=('rating', 'mean'))  # 聚合评论特征。
customer_features = customers.set_index('customer_id').join(order_features).join(review_features)  # 以客户主表左连接两类特征。
customer_features[['completed_orders', 'total_spend', 'review_count']] = customer_features[['completed_orders', 'total_spend', 'review_count']].fillna(0)  # 无行为客户的计数金额填零。
customer_features['days_as_customer'] = (pd.Timestamp('2025-03-31') - customer_features['signup_date']).dt.days  # 计算客户存续天数。
assert len(customer_features) == len(customers) and customer_features.index.is_unique and customer_features['total_spend'].ge(0).all()  # 验证客户粒度、唯一性和金额。
print(customer_features.sort_values('total_spend', ascending=False))  # 输出最终客户特征表。
