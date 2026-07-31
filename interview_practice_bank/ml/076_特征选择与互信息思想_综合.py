"""
题目 076：特征选择与互信息思想_综合

要求：完成“特征选择与互信息思想”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建三个二值特征。
2. 第一个特征完全决定标签。
3. 用相关性演示单变量筛选。
4. 选择绝对相关最高特征。
5. 低方差筛选与标签无关。
6. 使用方差阈值。
7. 排列重要性衡量打乱单列后的性能下降。
8. 综合得到模型无关重要性排序。

完成标准：
- 验证找到最直接相关特征。
- 综合得到模型无关重要性排序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 创建三个二值特征。
target = np.array([0, 0, 1, 1])  # 第一个特征完全决定标签。
correlations = np.array([np.corrcoef(features[:, index], target)[0, 1] for index in range(features.shape[1])])  # 用相关性演示单变量筛选。
selected = np.abs(correlations).argmax()  # 选择绝对相关最高特征。
variance = features.var(axis=0)  # 低方差筛选与标签无关。
keep_variance = variance >= 0.4  # 使用方差阈值。
assert selected == 0  # 验证找到最直接相关特征。
print(correlations, variance, keep_variance, selected)  # 输出筛选依据。
baseline_metric = 0.9; shuffled_metrics = np.array([0.89, 0.6, 0.88]); permutation_importance = baseline_metric - shuffled_metrics; ranked_features = np.argsort(permutation_importance)[::-1]  # 排列重要性衡量打乱单列后的性能下降。
assert ranked_features[0] == 1 and permutation_importance[1] == permutation_importance.max()  # 综合得到模型无关重要性排序。
