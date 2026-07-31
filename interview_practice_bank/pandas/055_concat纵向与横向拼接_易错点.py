"""
题目 055：concat纵向与横向拼接_易错点

要求：完成“concat纵向与横向拼接”的易错点题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
first = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}).set_index('id')  # 创建第一块索引数据。
second = pd.DataFrame({'id': [3, 4], 'value': [30, 40]}).set_index('id')  # 创建第二块索引数据。
rows = pd.concat([first, second], axis=0, verify_integrity=True)  # 纵向拼接并检查索引冲突。
labels = pd.Series({1: 'A', 2: 'B', 3: 'C', 4: 'D'}, name='label')  # 创建同索引标签列。
columns = pd.concat([rows, labels], axis=1, join='inner')  # 横向按索引对齐。
assert columns.shape == (4, 2)  # 验证拼接shape。
print(columns)  # 输出拼接结果。
