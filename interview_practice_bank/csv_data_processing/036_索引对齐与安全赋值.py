"""
CSV数据处理练习 036：索引对齐与安全赋值

题目：读取客户CSV，设置customer_id索引，并把乱序的风险分数按标签自动对齐。

操作过程：
1. 定位客户CSV。
2. 使用稳定业务键作为索引。
3. 创建顺序不同且只覆盖部分客户的Series。
4. 按索引标签对齐赋值而非按位置。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'customers.csv'  # 定位客户CSV。
frame = pd.read_csv(csv_path).set_index('customer_id')  # 使用稳定业务键作为索引。
risk = pd.Series({'C003': 0.8, 'C001': 0.2, 'C002': 0.5}, name='risk')  # 创建顺序不同且只覆盖部分客户的Series。
frame.loc[:, 'risk'] = risk  # 按索引标签对齐赋值而非按位置。
assert frame.loc['C003', 'risk'] == 0.8 and frame['risk'].isna().sum() == len(frame) - len(risk)  # 验证标签对齐和未匹配缺失。
print(frame[['customer_name', 'risk']])  # 输出对齐结果。
