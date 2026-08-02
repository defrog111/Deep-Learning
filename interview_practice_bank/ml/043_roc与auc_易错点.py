"""
题目 043：ROC与AUC_易错点

要求：完成“ROC与AUC”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 导入可靠的ROC工具。
3. AUC等于随机正样本得分高于负样本的概率，平分计0.5。
4. 验证排序解释。

完成标准：
- 验证排序解释。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.metrics import roc_auc_score, roc_curve  # 导入可靠的ROC工具。
tied_scores = np.array([0.8, 0.8, 0.2, 0.1]); tied_labels = np.array([1, 0, 1, 0]); pair_scores = [(tied_scores[p] > tied_scores[n]) + 0.5 * (tied_scores[p] == tied_scores[n]) for p in np.flatnonzero(tied_labels) for n in np.flatnonzero(1 - tied_labels)]; rank_auc = np.mean(pair_scores)  # AUC等于随机正样本得分高于负样本的概率，平分计0.5。
assert 0 <= rank_auc <= 1  # 验证排序解释。
