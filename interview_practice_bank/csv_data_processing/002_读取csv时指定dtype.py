"""
CSV数据处理练习 002：读取CSV时指定dtype

题目：读取员工CSV时为ID使用可空整数、部门使用category，并比较转换后的内存和类型。

操作过程：
1. 定位员工CSV。
2. 在读取入口指定稳定dtype。
3. 计算包含字符串内容的内存占用。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'employees.csv'  # 定位员工CSV。
frame = pd.read_csv(csv_path, dtype={'employee_id': 'Int64', 'manager_id': 'Int64', 'department': 'category'})  # 在读取入口指定稳定dtype。
memory_bytes = frame.memory_usage(deep=True).sum()  # 计算包含字符串内容的内存占用。
assert str(frame['manager_id'].dtype) == 'Int64' and str(frame['department'].dtype) == 'category'  # 验证可空整数和分类类型。
print(frame.dtypes, memory_bytes, sep='\n')  # 输出类型和内存。
