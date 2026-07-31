"""
题目 022：数据类型转换_变式

要求：完成“数据类型转换”的变式题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造脏类型数据。
2. 非法数字转缺失并使用可空整数。
3. 显式映射布尔字符串。
4. 低基数文本转分类类型。

完成标准：
- 验证可空整数类型。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'id': ['1', '2', 'bad'], 'active': ['True', 'False', 'True'], 'group': ['x', 'y', 'x']})  # 构造脏类型数据。
frame['id'] = pd.to_numeric(frame['id'], errors='coerce').astype('Int64')  # 非法数字转缺失并使用可空整数。
frame['active'] = frame['active'].map({'True': True, 'False': False}).astype('boolean')  # 显式映射布尔字符串。
frame['group'] = frame['group'].astype('category')  # 低基数文本转分类类型。
assert str(frame['id'].dtype) == 'Int64'  # 验证可空整数类型。
print(frame.dtypes, '\n', frame)  # 输出类型和数据。
