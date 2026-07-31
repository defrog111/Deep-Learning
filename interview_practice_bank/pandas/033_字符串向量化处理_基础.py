"""
题目 033：字符串向量化处理_基础

要求：完成“字符串向量化处理”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series(['  Alice-Smith ', 'bob-jones', None], dtype='string')  # 使用Pandas字符串类型并保留缺失。
clean = series.str.strip().str.lower().str.replace('-', ' ', regex=False)  # 链式向量化清洗字符串。
parts = clean.str.extract(r'(?P<first>\w+)\s+(?P<last>\w+)')  # 用正则提取命名分组。
contains_o = clean.str.contains('o', na=False)  # 缺失值按False处理。
assert parts.columns.tolist() == ['first', 'last']  # 验证提取结果列名。
print(clean, '\n', parts, '\n', contains_o)  # 输出清洗和提取结果。
