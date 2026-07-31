"""
题目 044：类别权重与不平衡_变式

要求：完成“类别权重与不平衡”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.linear_model import LogisticRegression  # 导入分类器。
from sklearn.metrics import balanced_accuracy_score  # 导入平衡准确率。
rng = np.random.default_rng(42)  # 固定随机数。
features = rng.normal(size=(110, 3))  # 创建特征。
labels = np.array([0] * 100 + [1] * 10)  # 创建严重不平衡标签。
features[labels == 1, 0] += 2  # 为少数类加入可学习信号。
model = LogisticRegression(class_weight='balanced', C=2).fit(features, labels)  # 自动按频次设置反比类别权重。
predictions = model.predict(features)  # 生成预测。
score = balanced_accuracy_score(labels, predictions)  # 各类召回率等权平均。
assert score >= 0.5  # 至少不差于随机基准。
print(model.class_weight, score, np.bincount(predictions, minlength=2))  # 输出不平衡结果。
