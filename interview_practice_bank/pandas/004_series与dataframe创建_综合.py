"""
题目 004：Series与DataFrame创建_综合

要求：完成“Series与DataFrame创建”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建字符串扩展类型列。
2. 创建允许缺失的整数列。
3. 创建带完整类别集合的分类数据。
4. 横向拼接Series并加入分类列。
5. 对可空比较结果进行明确缺失处理。
6. 设置业务索引并补充缺失成员。

完成标准：
- 验证综合dtype设计。
- 验证重建索引后的shape和缺失行。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
names = pd.Series(['Ada', 'Bob', 'Cyd'], name='name', dtype='string')  # 创建字符串扩展类型列。
scores = pd.Series([91, None, 85], name='score', dtype='Int64')  # 创建允许缺失的整数列。
teams = pd.Categorical(['A', 'B', 'A'], categories=['A', 'B', 'C'])  # 创建带完整类别集合的分类数据。
frame = pd.concat([names, scores], axis=1).assign(team=teams)  # 横向拼接Series并加入分类列。
frame['passed'] = frame['score'].ge(80).fillna(False)  # 对可空比较结果进行明确缺失处理。
summary = frame.set_index('name').reindex(['Ada', 'Bob', 'Cyd', 'Dan'])  # 设置业务索引并补充缺失成员。
assert str(frame['name'].dtype) == 'string' and str(frame['score'].dtype) == 'Int64'  # 验证综合dtype设计。
assert summary.shape == (4, 3) and summary.loc['Dan'].isna().all()  # 验证重建索引后的shape和缺失行。
print(frame, frame.dtypes, summary)  # 输出综合创建和索引结果。
