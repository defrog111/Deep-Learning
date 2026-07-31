"""
题目 075：特征选择与互信息思想_易错点

要求：完成“特征选择与互信息思想”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建三个二值特征。
2. 第一个特征完全决定标签。
3. 用相关性演示单变量筛选。
4. 选择绝对相关最高特征。
5. 低方差筛选与标签无关。
6. 使用方差阈值。
7. wrapper方法必须在验证或CV内部选特征数。
8. 防止按训练分数选择全部特征。

完成标准：
- 验证找到最直接相关特征。
- 防止按训练分数选择全部特征。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 创建三个二值特征。
target = np.array([0, 0, 1, 1])  # 第一个特征完全决定标签。
correlations = np.array([np.corrcoef(features[:, index], target)[0, 1] for index in range(features.shape[1])])  # 用相关性演示单变量筛选。
selected = np.abs(correlations).argmax()  # 选择绝对相关最高特征。
variance = features.var(axis=0)  # 低方差筛选与标签无关。
keep_variance = variance >= 0.30000000000000004  # 使用方差阈值。
assert selected == 0  # 验证找到最直接相关特征。
print(correlations, variance, keep_variance, selected)  # 输出筛选依据。
selection_train_scores = np.array([0.8, 0.9, 1.0]); selection_validation_scores = np.array([0.78, 0.85, 0.7]); selected_feature_count = np.array([1, 2, 3])[selection_validation_scores.argmax()]  # wrapper方法必须在验证或CV内部选特征数。
assert selected_feature_count == 2  # 防止按训练分数选择全部特征。
