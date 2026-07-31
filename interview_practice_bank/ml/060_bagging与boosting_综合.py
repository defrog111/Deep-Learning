"""
题目 060：Bagging与Boosting_综合

要求：完成“Bagging与Boosting”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 三个基学习器的分类结果。
2. Bagging并行模型多数投票降低方差。
3. 构造Boosting弱学习器输出。
4. 后续学习器按性能加权。
5. Boosting串行加权纠错。
6. Stacking元学习器必须训练在out-of-fold预测上。
7. 综合验证两层模型输入。

完成标准：
- 验证集成输出shape。
- 综合验证两层模型输入。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
predictions = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]])  # 三个基学习器的分类结果。
bagging_vote = (predictions.mean(axis=0) >= 0.5).astype(int)  # Bagging并行模型多数投票降低方差。
weak_outputs = np.array([0.2, -0.4, 0.7])  # 构造Boosting弱学习器输出。
weights = np.array([0.2, 0.3, 0.6000000000000001])  # 后续学习器按性能加权。
boosting_score = np.dot(weights, weak_outputs)  # Boosting串行加权纠错。
assert bagging_vote.shape == (4,)  # 验证集成输出shape。
print(bagging_vote, boosting_score)  # 输出两类集成思想结果。
base_oof = np.array([[0.8, 0.6], [0.2, 0.3], [0.7, 0.9], [0.1, 0.4]]); meta_weights = np.linalg.lstsq(np.c_[np.ones(4), base_oof], np.array([1, 0, 1, 0]), rcond=None)[0]; stacked = np.c_[np.ones(4), base_oof] @ meta_weights  # Stacking元学习器必须训练在out-of-fold预测上。
assert stacked.shape == (4,) and np.isfinite(stacked).all()  # 综合验证两层模型输入。
