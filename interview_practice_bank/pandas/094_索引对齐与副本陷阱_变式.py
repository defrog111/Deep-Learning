"""
题目 094：索引对齐与副本陷阱_变式

要求：完成“索引对齐与副本陷阱”的变式题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
left = pd.Series([10, 20], index=['a', 'b'])  # 创建第一组带标签数据。
right = pd.Series([1, 2], index=['b', 'c'])  # 创建错位索引数据。
aligned_sum = left.add(right, fill_value=0)  # 显式按索引对齐并填充缺失。
frame = pd.DataFrame({'value': [1, 2, 3]})  # 构造用于安全修改的表。
selected = frame.loc[frame['value'].gt(1)].copy()  # 显式copy避免链式赋值歧义。
selected.loc[:, 'double'] = selected['value'] * 2  # 使用loc安全赋值。
assert aligned_sum.index.tolist() == ['a', 'b', 'c']  # 验证索引并集。
print(aligned_sum, '\n', selected)  # 输出对齐和副本结果。
