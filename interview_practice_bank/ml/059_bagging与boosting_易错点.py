"""
题目 059：Bagging与Boosting_易错点

要求：完成“Bagging与Boosting”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. AdaBoost提高错分样本权重。
3. 验证错分样本被关注。

完成标准：
- 验证错分样本被关注。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
boost_labels = np.array([1, 1, -1, -1]); weak_predictions = np.array([1, -1, -1, -1]); sample_weights_boost = np.full(4, 0.25); weighted_error = sample_weights_boost[weak_predictions != boost_labels].sum(); learner_weight = 0.5 * np.log((1 - weighted_error) / weighted_error); updated_weights = sample_weights_boost * np.exp(-learner_weight * boost_labels * weak_predictions); updated_weights /= updated_weights.sum()  # AdaBoost提高错分样本权重。
assert updated_weights[1] == updated_weights.max()  # 验证错分样本被关注。
