"""
题目 064：melt与宽长表转换_综合

要求：完成“melt与宽长表转换”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
wide = pd.DataFrame({'id': [1, 2], 'math': [90, 80], 'english': [85, 88]})  # 构造宽表。
long = wide.melt(id_vars='id', var_name='subject', value_name='score')  # 宽表转长表。
restored = long.pivot(index='id', columns='subject', values='score').reset_index()  # 长表恢复宽表。
restored.columns.name = None  # 清除列索引名称便于比较。
assert restored[['id', 'math', 'english']].equals(wide)  # 验证往返转换保持数据。
print(long, '\n', restored)  # 输出两种形态。
