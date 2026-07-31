"""
题目 030：重复值识别与删除_变式

要求：完成“重复值识别与删除”的变式题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'id': [1, 1, 2, 2], 'value': [10, 12, 20, 20], 'time': [1, 2, 1, 1]})  # 构造业务主键重复数据。
duplicate_mask = frame.duplicated(subset=['id'], keep=False)  # 标出所有重复业务键。
latest = frame.sort_values('time').drop_duplicates('id', keep='last')  # 每个id保留最新记录。
exact_unique = frame.drop_duplicates()  # 删除整行完全重复记录。
assert latest['id'].is_unique  # 验证业务键唯一。
print(frame[duplicate_mask], '\n', latest, '\n', exact_unique)  # 对比不同去重策略。
