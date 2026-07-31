"""
题目 026：KNeighborsClassifier_变式

要求：完成“KNeighborsClassifier”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.model_selection import cross_val_score  # 导入交叉验证。
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # KNN对尺度敏感需标准化。
features, labels = load_iris(return_X_y=True)  # 加载数据。
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))  # 配置距离加权KNN。
scores = cross_val_score(model, features, labels, cv=5)  # 交叉验证泛化性能。
assert scores.mean() > 0.85  # 验证模型表现。
print(scores, scores.mean())  # 输出各折分数。
