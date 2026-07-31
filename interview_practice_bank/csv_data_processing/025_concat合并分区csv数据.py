"""
CSV数据处理练习 025：concat合并分区CSV数据

题目：读取销售CSV，模拟按月份分区后重新纵向拼接，并用keys保留来源。

操作过程：
1. 定位销售CSV。
2. 读取并解析日期。
3. 模拟按月文件分区。
4. 用字典keys生成来源层级索引。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path, parse_dates=['date'])  # 读取并解析日期。
partitions = {str(month): group.copy() for month, group in frame.groupby(frame['date'].dt.to_period('M'))}  # 模拟按月文件分区。
combined = pd.concat(partitions, names=['source_month', 'row'])  # 用字典keys生成来源层级索引。
assert len(combined) == len(frame) and combined.index.names == ['source_month', 'row']  # 验证行数和来源索引。
print(combined[['order_id', 'date']].head())  # 输出带来源的拼接结果。
