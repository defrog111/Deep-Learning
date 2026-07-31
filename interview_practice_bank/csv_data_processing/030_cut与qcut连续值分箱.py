"""
CSV数据处理练习 030：cut与qcut连续值分箱

题目：读取客户CSV，分别用业务固定边界和等频分位数对年龄分箱。

操作过程：
1. 定位客户CSV。
2. 读取客户年龄。
3. 按业务边界分箱。
4. 按样本数量近似等分。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'customers.csv'  # 定位客户CSV。
frame = pd.read_csv(csv_path)  # 读取客户年龄。
frame['age_band'] = pd.cut(frame['age'], bins=[0, 29, 39, 49, float('inf')], labels=['under30', '30s', '40s', '50plus'], include_lowest=True)  # 按业务边界分箱。
frame['age_quantile'] = pd.qcut(frame['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # 按样本数量近似等分。
assert not frame[['age_band', 'age_quantile']].isna().any().any()  # 验证所有年龄落入分箱。
print(frame[['customer_name', 'age', 'age_band', 'age_quantile']])  # 对比两种分箱。
