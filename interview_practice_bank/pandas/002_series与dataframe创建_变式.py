"""
题目 002：Series与DataFrame创建_变式

要求：完成“Series与DataFrame创建”的变式题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建键不完全一致的记录列表。
2. 用记录构造表并直接指定索引列。
3. 使用可空dtype和assign派生列。

完成标准：
- 验证索引名称和缺失值对齐。
- 验证Pandas扩展dtype。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
records = [{'name': 'Ada', 'score': 91}, {'name': 'Bob'}, {'name': 'Cyd', 'score': 85}]  # 创建键不完全一致的记录列表。
frame = pd.DataFrame.from_records(records, index='name')  # 用记录构造表并直接指定索引列。
frame = frame.assign(score=frame['score'].astype('Float64'), passed=lambda data: data['score'].ge(80))  # 使用可空dtype和assign派生列。
assert frame.index.name == 'name' and frame['score'].isna().sum() == 1  # 验证索引名称和缺失值对齐。
assert str(frame['score'].dtype) == 'Float64' and str(frame['passed'].dtype) == 'boolean'  # 验证Pandas扩展dtype。
print(frame, frame.dtypes)  # 输出记录式构造结果。
