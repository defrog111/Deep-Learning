"""
题目 020：缺失值检测与填充_综合

要求：完成“缺失值检测与填充”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造含缺失值的数据。
2. 按列统计缺失值。
3. 类别缺失值使用明确标签填充。
4. 先组内中位数再兜底。
5. 对比处理前统计与处理后数据。

完成标准：
- 验证所有缺失值均已处理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': ['A', 'A', 'B', None], 'score': [10.0, None, 30.0, None]})  # 构造含缺失值的数据。
missing_count = frame.isna().sum()  # 按列统计缺失值。
frame['team'] = frame['team'].fillna('Unknown')  # 类别缺失值使用明确标签填充。
frame['score'] = frame['score'].fillna(frame.groupby('team')['score'].transform('median')).fillna(0)  # 先组内中位数再兜底。
assert not frame.isna().any().any()  # 验证所有缺失值均已处理。
print(missing_count, '\n', frame)  # 对比处理前统计与处理后数据。
