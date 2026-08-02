"""
题目 042：ROC与AUC_变式

要求：完成“ROC与AUC”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 导入可靠的ROC工具。
3. PR-AUC在正类稀少时比ROC-AUC更关注正类质量。
4. 验证曲线面积范围。

完成标准：
- 验证曲线面积范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.metrics import roc_auc_score, roc_curve  # 导入可靠的ROC工具。
precision_points = np.array([1.0, 0.75, 0.5]); recall_points = np.array([0.0, 0.5, 1.0]); pr_auc = np.trapezoid(precision_points, recall_points)  # PR-AUC在正类稀少时比ROC-AUC更关注正类质量。
assert 0 <= pr_auc <= 1  # 验证曲线面积范围。
