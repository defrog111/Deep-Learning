"""
题目 029：RandomForest特征重要性_基础

要求：完成“RandomForest特征重要性”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林。
features, labels = load_iris(return_X_y=True)  # 加载数据。
model = RandomForestClassifier(n_estimators=30, max_depth=4, oob_score=True, bootstrap=True, random_state=42).fit(features, labels)  # 训练带袋外评估的森林。
importance = model.feature_importances_  # 取得基于不纯度的特征重要性。
order = np.argsort(importance)[::-1]  # 按重要性排序。
assert np.isclose(importance.sum(), 1) and model.oob_score_ > 0.8  # 验证重要性归一化和OOB表现。
print(order, importance, model.oob_score_)  # 输出特征排名。
