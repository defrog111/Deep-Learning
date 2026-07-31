"""
题目 042：ROC_AUC与PR_AUC_变式

要求：完成“ROC_AUC与PR_AUC”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入分类器。
from sklearn.metrics import average_precision_score, roc_auc_score  # 导入排序指标。
from sklearn.model_selection import cross_val_predict  # 导入交叉验证概率预测。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
probabilities = cross_val_predict(RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42), features, labels, cv=3, method='predict_proba')[:, 1]  # 获取未见样本概率。
roc_auc = roc_auc_score(labels, probabilities)  # 计算ROC AUC。
pr_auc = average_precision_score(labels, probabilities)  # 计算PR曲线平均精度。
assert roc_auc > 0.9 and pr_auc > 0.9  # 验证模型排序能力。
print(roc_auc, pr_auc)  # 输出两种AUC。
