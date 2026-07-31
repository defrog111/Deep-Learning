"""
题目 026：排序排名与TopN_变式

要求：完成“排序排名与TopN”的变式题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造含并列分数的数据。
2. 计算无跳号排名。
3. 使用nlargest高效取得前N名。
4. 多列稳定排序。

完成标准：
- 验证最高分位于首行。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'name': list('ABCDE'), 'score': [88, 95, 88, 72, 91]})  # 构造含并列分数的数据。
frame['dense_rank'] = frame['score'].rank(method='dense', ascending=False).astype(int)  # 计算无跳号排名。
top = frame.nlargest(3, 'score')  # 使用nlargest高效取得前N名。
sorted_frame = frame.sort_values(['score', 'name'], ascending=[False, True])  # 多列稳定排序。
assert sorted_frame.iloc[0]['score'] == frame['score'].max()  # 验证最高分位于首行。
print(top, '\n', sorted_frame)  # 输出TopN与完整排序。
