"""
CSV数据处理练习 035：Categorical有序类别

题目：读取脏订单CSV，清洗状态并建立有业务顺序的Categorical类型。

操作过程：
1. 定位订单CSV。
2. 读取订单状态。
3. 标准化状态文本。
4. 定义业务顺序。
5. 转换为有序分类数据。
6. 按类别顺序而非字母排序。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位订单CSV。
frame = pd.read_csv(csv_path)  # 读取订单状态。
clean_status = frame['status'].str.strip().str.lower()  # 标准化状态文本。
status_type = pd.CategoricalDtype(categories=['pending', 'completed', 'returned', 'cancelled'], ordered=True)  # 定义业务顺序。
frame['status_category'] = clean_status.astype(status_type)  # 转换为有序分类数据。
sorted_orders = frame.sort_values('status_category')  # 按类别顺序而非字母排序。
assert str(frame['status_category'].dtype) == 'category' and frame['status_category'].cat.ordered  # 验证有序分类类型。
print(sorted_orders[['order_id', 'status_category']])  # 输出业务顺序结果。
