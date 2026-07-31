"""
题目 044：ROC与AUC_综合

要求：完成“ROC与AUC”的综合题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.metrics import roc_auc_score, roc_curve  # 导入可靠的ROC工具。
targets = np.array([0, 0, 1, 1])  # 创建真实标签。
scores = np.array([0.1, 0.4, 0.35, 0.8])  # 创建连续正类分数。
false_positive_rate, true_positive_rate, thresholds = roc_curve(targets, scores)  # 扫描所有阈值得到ROC点。
auc = roc_auc_score(targets, scores)  # AUC等价于随机正样本得分高于负样本的概率。
assert np.isclose(auc, 0.75)  # 验证经典例子AUC。
print(false_positive_rate, true_positive_rate, thresholds, auc)  # 输出ROC曲线数据。
