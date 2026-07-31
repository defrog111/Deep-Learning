"""
题目 092：cut与qcut分箱_综合

要求：完成“cut与qcut分箱”的综合题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
scores = pd.Series([45, 58, 67, 72, 81, 89, 95, 99], name='score')  # 构造分数序列。
fixed = pd.cut(scores, bins=[0, 60, 80, 100], labels=['low', 'medium', 'high'], right=False)  # 按业务边界等距分箱。
quantiles = pd.qcut(scores, q=4, labels=False, duplicates='drop')  # 按样本分位数等频分箱。
table = pd.DataFrame({'score': scores, 'fixed': fixed, 'quantile': quantiles})  # 汇总两种分箱。
assert table['fixed'].notna().all()  # 验证边界覆盖所有值。
print(table)  # 输出分箱结果。
