"""
CSV数据处理练习 001：读取CSV并检查结构

题目：读取销售CSV，依次检查shape、列名、dtype、前两行和缺失值数量。

操作过程：
1. 定位销售CSV。
2. 从CSV读取完整表格。
3. 按列统计缺失值。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取完整表格。
missing = frame.isna().sum()  # 按列统计缺失值。
assert frame.shape[1] == 9 and frame.columns[0] == 'order_id'  # 验证列数和首列名称。
print(frame.shape, frame.dtypes, frame.head(2), missing, sep='\n')  # 输出结构检查结果。
