"""
题目 074：特征选择与互信息思想_变式

要求：完成“特征选择与互信息思想”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建三个二值特征。
2. 第一个特征完全决定标签。
3. 用相关性演示单变量筛选。
4. 选择绝对相关最高特征。
5. 低方差筛选与标签无关。
6. 使用方差阈值。
7. 过滤法可先删除高度共线特征。

完成标准：
- 验证找到最直接相关特征。
- 验证冗余特征识别。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 创建三个二值特征。
target = np.array([0, 0, 1, 1])  # 第一个特征完全决定标签。
correlations = np.array([np.corrcoef(features[:, index], target)[0, 1] for index in range(features.shape[1])])  # 用相关性演示单变量筛选。
selected = np.abs(correlations).argmax()  # 选择绝对相关最高特征。
variance = features.var(axis=0)  # 低方差筛选与标签无关。
keep_variance = variance >= 0.2  # 使用方差阈值。
assert selected == 0  # 验证找到最直接相关特征。
print(correlations, variance, keep_variance, selected)  # 输出筛选依据。
feature_matrix = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 1]], dtype=float); feature_correlation = np.corrcoef(feature_matrix, rowvar=False); redundant_pair = abs(feature_correlation[0, 1]) > 0.95  # 过滤法可先删除高度共线特征。
assert redundant_pair  # 验证冗余特征识别。
