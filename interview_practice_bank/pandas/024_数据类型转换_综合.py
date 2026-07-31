"""
题目 024：数据类型转换_综合

要求：完成“数据类型转换”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'id': ['1', '2', 'bad'], 'active': ['True', 'False', 'True'], 'group': ['x', 'y', 'x']})  # 构造脏类型数据。
frame['id'] = pd.to_numeric(frame['id'], errors='coerce').astype('Int64')  # 非法数字转缺失并使用可空整数。
frame['active'] = frame['active'].map({'True': True, 'False': False}).astype('boolean')  # 显式映射布尔字符串。
frame['group'] = frame['group'].astype('category')  # 低基数文本转分类类型。
assert str(frame['id'].dtype) == 'Int64'  # 验证可空整数类型。
print(frame.dtypes, '\n', frame)  # 输出类型和数据。
