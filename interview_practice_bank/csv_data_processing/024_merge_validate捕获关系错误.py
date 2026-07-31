"""
CSV数据处理练习 024：merge validate捕获关系错误

题目：读取客户和原始脏订单CSV，演示错误的一对一假设会被validate拒绝。

操作过程：
1. 定位共享数据目录。
2. 读取唯一客户。
3. 保留重复订单以演示错误。
4. 尝试声明不成立的一对一关系。
5. customer_id在订单侧会重复。
6. 保存清晰错误信息。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
data_folder = Path(__file__).parents[1] / 'data'  # 定位共享数据目录。
customers = pd.read_csv(data_folder / 'customers.csv')  # 读取唯一客户。
orders = pd.read_csv(data_folder / 'orders_dirty.csv')  # 保留重复订单以演示错误。
try:  # 尝试声明不成立的一对一关系。
    orders.merge(customers, on='customer_id', validate='one_to_one')  # customer_id在订单侧会重复。
except pd.errors.MergeError as error:  # 捕获关系验证错误。
    merge_error = str(error)  # 保存清晰错误信息。
assert 'not a one-to-one' in merge_error  # 验证validate阻止静默行数膨胀。
print(merge_error)  # 输出关系错误说明。
