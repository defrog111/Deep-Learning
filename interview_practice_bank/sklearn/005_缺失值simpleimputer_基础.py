"""
题目 005：缺失值SimpleImputer_基础

要求：完成“缺失值SimpleImputer”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建含NaN训练数据。
2. 创建含缺失测试样本。
3. 只用训练数据拟合填充值。
4. 填充训练集。
5. 用训练填充值处理测试集。

完成标准：
- 验证无剩余NaN。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.impute import SimpleImputer  # 导入缺失值填充器。
train = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 8.0]])  # 创建含NaN训练数据。
test = np.array([[np.nan, 100.0]])  # 创建含缺失测试样本。
imputer = SimpleImputer(strategy='median').fit(train)  # 只用训练数据拟合填充值。
filled_train = imputer.transform(train)  # 填充训练集。
filled_test = imputer.transform(test)  # 用训练填充值处理测试集。
assert not np.isnan(filled_train).any() and not np.isnan(filled_test).any()  # 验证无剩余NaN。
print(imputer.statistics_, filled_test)  # 输出填充值和结果。
