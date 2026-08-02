"""
题目 020：逻辑回归与Sigmoid_综合

要求：完成“逻辑回归与Sigmoid”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合用log概率实现朴素贝叶斯分类思想。
3. 验证概率模型计算。

完成标准：
- 验证概率模型计算。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
log_prob_feature = np.array([[np.log(0.8), np.log(0.2)], [np.log(0.3), np.log(0.7)]]); log_prior = np.log(np.array([0.5, 0.5])); naive_bayes_scores = log_prob_feature.sum(axis=0) + log_prior; naive_bayes_prediction = naive_bayes_scores.argmax()  # 综合用log概率实现朴素贝叶斯分类思想。
assert naive_bayes_prediction in (0, 1) and np.isfinite(naive_bayes_scores).all()  # 验证概率模型计算。
