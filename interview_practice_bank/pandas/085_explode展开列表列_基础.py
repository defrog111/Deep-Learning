"""
题目 085：explode展开列表列_基础

要求：完成“explode展开列表列”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'order': [1, 2], 'items': [['pen', 'book'], ['desk']], 'quantities': [[2, 1], [1]]})  # 构造列表列。
long = frame.explode(['items', 'quantities'], ignore_index=True)  # 同时展开等长列表列。
long['quantities'] = long['quantities'].astype(int)  # explode后恢复数值类型。
assert len(long) == 3  # 验证一行多值已展开。
print(long)  # 输出规范化长表。
