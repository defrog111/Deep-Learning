"""
题目 044：ROC与AUC_综合

要求：完成“ROC与AUC”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 导入可靠的ROC工具。
3. 多分类One-vs-Rest可使用macro或weighted汇总。
4. 综合显示多数类表现会主导weighted指标。

完成标准：
- 综合显示多数类表现会主导weighted指标。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.metrics import roc_auc_score, roc_curve  # 导入可靠的ROC工具。
class_aucs = np.array([0.95, 0.7, 0.6]); class_supports = np.array([80, 15, 5]); macro_auc = class_aucs.mean(); weighted_auc = np.average(class_aucs, weights=class_supports)  # 多分类One-vs-Rest可使用macro或weighted汇总。
assert weighted_auc > macro_auc  # 综合显示多数类表现会主导weighted指标。
