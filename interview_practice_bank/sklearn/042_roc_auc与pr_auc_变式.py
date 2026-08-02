"""
题目 042：ROC_AUC与PR_AUC_变式

要求：完成“ROC_AUC与PR_AUC”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 曲线点数量通常比阈值多1。

完成标准：
- 验证PR和ROC返回结构。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入分类器。
from sklearn.metrics import average_precision_score, roc_auc_score  # 导入排序指标。
from sklearn.model_selection import cross_val_predict  # 导入交叉验证概率预测。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.metrics import precision_recall_curve, roc_curve  # 导入阈值曲线。
binary_labels = (labels == labels.max()).astype(int); binary_scores = np.linspace(0, 1, len(labels)); precision_curve, recall_curve, pr_thresholds = precision_recall_curve(binary_labels, binary_scores); false_positive, true_positive, roc_thresholds = roc_curve(binary_labels, binary_scores)  # 曲线点数量通常比阈值多1。
assert len(precision_curve) == len(pr_thresholds) + 1 and len(false_positive) == len(roc_thresholds) and np.all(np.diff(false_positive) >= 0)  # 验证PR和ROC返回结构。
