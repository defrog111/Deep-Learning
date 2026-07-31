"""
题目 051：merge多表连接_易错点

要求：完成“merge多表连接”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 定位数据目录。
2. 读取事实表。
3. 读取员工维表。
4. 多对一左连接并验证关系。
5. 防止连接丢行或膨胀。

完成标准：
- 防止连接丢行或膨胀。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入 Pandas。
base = Path(__file__).parents[1] / 'data'  # 定位数据目录。
sales = pd.read_csv(base / 'sales.csv')  # 读取事实表。
employees = pd.read_csv(base / 'employees.csv')  # 读取员工维表。
answer = sales.merge(employees[['name', 'department', 'city']], left_on='salesperson', right_on='name', how='left', validate='many_to_one', indicator=True)  # 多对一左连接并验证关系。
assert answer['_merge'].eq('both').all() and len(answer) == len(sales)  # 防止连接丢行或膨胀。
print(answer[['order_id', 'salesperson', 'department', 'city']].head())  # 查看连接结果。
