"""
题目 090：cut与qcut分箱_变式

要求：完成“cut与qcut分箱”的变式题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造分数序列。
2. 按业务边界等距分箱。
3. 按样本分位数等频分箱。
4. 汇总两种分箱。

完成标准：
- 验证边界覆盖所有值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
scores = pd.Series([45, 58, 67, 72, 81, 89, 95, 99], name='score')  # 构造分数序列。
fixed = pd.cut(scores, bins=[0, 60, 80, 100], labels=['low', 'medium', 'high'], right=False)  # 按业务边界等距分箱。
quantiles = pd.qcut(scores, q=4, labels=False, duplicates='drop')  # 按样本分位数等频分箱。
table = pd.DataFrame({'score': scores, 'fixed': fixed, 'quantile': quantiles})  # 汇总两种分箱。
assert table['fixed'].notna().all()  # 验证边界覆盖所有值。
print(table)  # 输出分箱结果。
