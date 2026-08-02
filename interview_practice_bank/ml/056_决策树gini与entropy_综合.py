"""
题目 056：决策树Gini与Entropy_综合

要求：完成“决策树Gini与Entropy”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合叶节点概率、不纯度和预测。
3. 两种指标量纲不同但纯度趋势一致。

完成标准：
- 两种指标量纲不同但纯度趋势一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
leaf_probabilities = np.array([0.7, 0.3]); leaf_gini = 1 - np.sum(leaf_probabilities**2); leaf_entropy = -np.sum(leaf_probabilities * np.log2(leaf_probabilities)); leaf_prediction = leaf_probabilities.argmax()  # 综合叶节点概率、不纯度和预测。
assert leaf_gini < leaf_entropy and leaf_prediction == 0  # 两种指标量纲不同但纯度趋势一致。
