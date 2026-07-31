"""
题目 059：Bagging与Boosting_易错点

要求：完成“Bagging与Boosting”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 三个基学习器的分类结果。
2. Bagging并行模型多数投票降低方差。
3. 构造Boosting弱学习器输出。
4. 后续学习器按性能加权。
5. Boosting串行加权纠错。
6. AdaBoost提高错分样本权重。

完成标准：
- 验证集成输出shape。
- 验证错分样本被关注。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
predictions = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]])  # 三个基学习器的分类结果。
bagging_vote = (predictions.mean(axis=0) >= 0.5).astype(int)  # Bagging并行模型多数投票降低方差。
weak_outputs = np.array([0.2, -0.4, 0.7])  # 构造Boosting弱学习器输出。
weights = np.array([0.2, 0.3, 0.55])  # 后续学习器按性能加权。
boosting_score = np.dot(weights, weak_outputs)  # Boosting串行加权纠错。
assert bagging_vote.shape == (4,)  # 验证集成输出shape。
print(bagging_vote, boosting_score)  # 输出两类集成思想结果。
boost_labels = np.array([1, 1, -1, -1]); weak_predictions = np.array([1, -1, -1, -1]); sample_weights_boost = np.full(4, 0.25); weighted_error = sample_weights_boost[weak_predictions != boost_labels].sum(); learner_weight = 0.5 * np.log((1 - weighted_error) / weighted_error); updated_weights = sample_weights_boost * np.exp(-learner_weight * boost_labels * weak_predictions); updated_weights /= updated_weights.sum()  # AdaBoost提高错分样本权重。
assert updated_weights[1] == updated_weights.max()  # 验证错分样本被关注。
