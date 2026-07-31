"""
题目 080：map替换与apply_综合

要求：完成“map替换与apply”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'grade': ['A', 'B', 'C', 'A'], 'score': [95, 82, 70, 91]})  # 构造映射数据。
frame['points'] = frame['grade'].map({'A': 4, 'B': 3, 'C': 2})  # 一对一字典映射。
frame['bucket'] = frame['score'].apply(lambda value: 'high' if value >= 90 else 'regular')  # 对单列执行自定义函数。
frame['grade'] = frame['grade'].replace({'A': 'Excellent'})  # 替换指定值。
assert frame['points'].notna().all()  # 验证映射无遗漏。
print(frame)  # 输出转换结果。
