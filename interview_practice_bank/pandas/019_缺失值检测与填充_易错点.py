"""
题目 019：缺失值检测与填充_易错点

要求：完成“缺失值检测与填充”的易错点题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'team': ['A', 'A', 'B', None], 'score': [10.0, None, 30.0, None]})  # 构造含缺失值的数据。
missing_count = frame.isna().sum()  # 按列统计缺失值。
frame['team'] = frame['team'].fillna('Unknown')  # 类别缺失值使用明确标签填充。
frame['score'] = frame['score'].fillna(frame.groupby('team')['score'].transform('median')).fillna(0)  # 先组内中位数再兜底。
assert not frame.isna().any().any()  # 验证所有缺失值均已处理。
print(missing_count, '\n', frame)  # 对比处理前统计与处理后数据。
