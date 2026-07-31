"""
题目 093：索引对齐与副本陷阱_基础

要求：完成“索引对齐与副本陷阱”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建第一组带标签数据。
2. 创建错位索引数据。
3. 显式按索引对齐并填充缺失。
4. 构造用于安全修改的表。
5. 显式copy避免链式赋值歧义。
6. 使用loc安全赋值。

完成标准：
- 验证索引并集。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
