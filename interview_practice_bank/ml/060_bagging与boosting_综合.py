"""
题目 060：Bagging与Boosting_综合

要求：完成“Bagging与Boosting”的综合题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
predictions = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]])  # 三个基学习器的分类结果。
bagging_vote = (predictions.mean(axis=0) >= 0.5).astype(int)  # Bagging并行模型多数投票降低方差。
weak_outputs = np.array([0.2, -0.4, 0.7])  # 构造Boosting弱学习器输出。
weights = np.array([0.2, 0.3, 0.6000000000000001])  # 后续学习器按性能加权。
boosting_score = np.dot(weights, weak_outputs)  # Boosting串行加权纠错。
assert bagging_vote.shape == (4,)  # 验证集成输出shape。
print(bagging_vote, boosting_score)  # 输出两类集成思想结果。
