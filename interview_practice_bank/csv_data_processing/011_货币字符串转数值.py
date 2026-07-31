"""
CSV数据处理练习 011：货币字符串转数值

题目：读取脏订单CSV，删除金额中的美元符号和千位逗号，再安全转换为数值。

操作过程：
1. 定位脏订单CSV。
2. 读取含货币字符串的订单。
3. 删除显示符号。
4. 非法或缺失金额转换为NaN。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path)  # 读取含货币字符串的订单。
clean_amount = frame['amount'].astype('string').str.replace('$', '', regex=False).str.replace(',', '', regex=False)  # 删除显示符号。
frame['amount_number'] = pd.to_numeric(clean_amount, errors='coerce')  # 非法或缺失金额转换为NaN。
assert frame['amount_number'].dtype.kind == 'f' and frame['amount_number'].max() == 1250  # 验证数值类型和千位金额。
print(frame[['amount', 'amount_number']])  # 对比原始与数值金额。
