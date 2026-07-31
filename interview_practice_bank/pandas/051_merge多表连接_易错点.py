"""
题目 051：merge多表连接_易错点

要求：完成“merge多表连接”的易错点题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入 Pandas。
base = Path(__file__).parents[1] / 'data'  # 定位数据目录。
sales = pd.read_csv(base / 'sales.csv')  # 读取事实表。
employees = pd.read_csv(base / 'employees.csv')  # 读取员工维表。
answer = sales.merge(employees[['name', 'department', 'city']], left_on='salesperson', right_on='name', how='left', validate='many_to_one', indicator=True)  # 多对一左连接并验证关系。
assert answer['_merge'].eq('both').all() and len(answer) == len(sales)  # 防止连接丢行或膨胀。
print(answer[['order_id', 'salesperson', 'department', 'city']].head())  # 查看连接结果。
