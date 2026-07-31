"""
题目 011：Pipeline防止数据泄漏_基础

要求：完成“Pipeline防止数据泄漏”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.model_selection import cross_val_score  # 导入交叉验证评分。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
features, labels = load_iris(return_X_y=True)  # 加载数据。
pipeline = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=500))])  # 把预处理放进每折内部防泄漏。
scores = cross_val_score(pipeline, features, labels, cv=4, scoring='accuracy')  # 对完整Pipeline交叉验证。
assert scores.mean() > 0.9  # 验证模型表现合理。
print(scores, scores.mean())  # 输出各折准确率。
