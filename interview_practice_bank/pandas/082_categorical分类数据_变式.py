"""
题目 082：Categorical分类数据_变式

要求：完成“Categorical分类数据”的变式题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
priority = pd.CategoricalDtype(['low', 'medium', 'high'], ordered=True)  # 定义有序分类类型。
frame = pd.DataFrame({'task': list('ABCD'), 'priority': ['high', 'low', 'medium', 'low']})  # 构造任务表。
frame['priority'] = frame['priority'].astype(priority)  # 转为有序Categorical。
sorted_frame = frame.sort_values('priority')  # 按业务顺序而非字母顺序排序。
codes = frame['priority'].cat.codes  # 获取内部整数编码。
assert sorted_frame.iloc[0]['priority'] == 'low'  # 验证排序顺序。
print(sorted_frame, '\n', codes)  # 输出分类排序和编码。
