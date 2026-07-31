"""
题目 008：读取CSV与基本检查_综合

要求：完成“读取CSV与基本检查”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入 Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位题库自带的销售 CSV。
frame = pd.read_csv(csv_path, parse_dates=['date'])  # 读取 CSV 并在入口解析日期。
summary = frame[['quantity', 'unit_price', 'discount']].describe()  # 计算数值列描述统计。
assert not frame.empty and frame['date'].dtype.kind == 'M'  # 验证数据非空且日期类型正确。
print(frame.head(3), '\n', summary)  # 查看前三行和统计摘要。
