"""
CSV数据处理练习 003：usecols与nrows选择读取

题目：只读取销售CSV的四列和前五行，减少不必要的IO与内存。

操作过程：
1. 定位销售CSV。
2. 定义真正需要的列。
3. 在解析阶段限制列和行。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
columns = ['order_id', 'region', 'quantity', 'unit_price']  # 定义真正需要的列。
frame = pd.read_csv(csv_path, usecols=columns, nrows=5)  # 在解析阶段限制列和行。
assert frame.shape == (5, 4) and set(frame.columns) == set(columns)  # 验证选择读取范围。
print(frame)  # 输出小范围数据。
