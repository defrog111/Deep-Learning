"""
题目 013：布尔筛选与query_基础

要求：完成“布尔筛选与query”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 读取销售数据。
2. 设置随变式变化的最低原价。
3. 向量化计算折扣前金额。
4. query中用@引用外部变量。

完成标准：
- 验证每一行都满足条件。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入 Pandas。
frame = pd.read_csv(Path(__file__).parents[1] / 'data' / 'sales.csv')  # 读取销售数据。
minimum = 300  # 设置随变式变化的最低原价。
frame['gross'] = frame['quantity'] * frame['unit_price']  # 向量化计算折扣前金额。
answer = frame.query('gross >= @minimum and category == "Electronics"')  # query中用@引用外部变量。
assert answer['gross'].ge(minimum).all()  # 验证每一行都满足条件。
print(answer[['order_id', 'product', 'gross']])  # 输出关键列。
