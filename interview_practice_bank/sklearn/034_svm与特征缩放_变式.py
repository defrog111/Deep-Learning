"""
题目 034：SVM与特征缩放_变式

要求：完成“SVM与特征缩放”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_breast_cancer  # 导入分类数据。
from sklearn.model_selection import cross_val_score  # 导入交叉验证。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # SVM对尺度敏感。
from sklearn.svm import SVC  # 导入核SVM。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
model = make_pipeline(StandardScaler(), SVC(C=2, kernel='rbf', probability=True, random_state=42))  # 标准化后训练RBF SVM。
scores = cross_val_score(model, features, labels, cv=3, scoring='roc_auc')  # 使用AUC交叉验证。
assert scores.mean() > 0.9  # 验证性能合理。
print(scores, scores.mean())  # 输出AUC。
