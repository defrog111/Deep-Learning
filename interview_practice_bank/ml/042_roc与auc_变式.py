"""
题目 042：ROC与AUC_变式

要求：完成“ROC与AUC”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建真实标签。
2. 创建连续正类分数。
3. 扫描所有阈值得到ROC点。
4. AUC等价于随机正样本得分高于负样本的概率。
5. PR-AUC在正类稀少时比ROC-AUC更关注正类质量。

完成标准：
- 验证经典例子AUC。
- 验证曲线面积范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.metrics import roc_auc_score, roc_curve  # 导入可靠的ROC工具。
targets = np.array([0, 0, 1, 1])  # 创建真实标签。
scores = np.array([0.1, 0.4, 0.35, 0.8])  # 创建连续正类分数。
false_positive_rate, true_positive_rate, thresholds = roc_curve(targets, scores)  # 扫描所有阈值得到ROC点。
auc = roc_auc_score(targets, scores)  # AUC等价于随机正样本得分高于负样本的概率。
assert np.isclose(auc, 0.75)  # 验证经典例子AUC。
print(false_positive_rate, true_positive_rate, thresholds, auc)  # 输出ROC曲线数据。
precision_points = np.array([1.0, 0.75, 0.5]); recall_points = np.array([0.0, 0.5, 1.0]); pr_auc = np.trapezoid(precision_points, recall_points)  # PR-AUC在正类稀少时比ROC-AUC更关注正类质量。
assert 0 <= pr_auc <= 1  # 验证曲线面积范围。
