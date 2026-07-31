"""
题目 025：排序排名与TopN_基础

要求：完成“排序排名与TopN”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'name': list('ABCDE'), 'score': [88, 95, 88, 72, 91]})  # 构造含并列分数的数据。
frame['dense_rank'] = frame['score'].rank(method='dense', ascending=False).astype(int)  # 计算无跳号排名。
top = frame.nlargest(2, 'score')  # 使用nlargest高效取得前N名。
sorted_frame = frame.sort_values(['score', 'name'], ascending=[False, True])  # 多列稳定排序。
assert sorted_frame.iloc[0]['score'] == frame['score'].max()  # 验证最高分位于首行。
print(top, '\n', sorted_frame)  # 输出TopN与完整排序。
