"""
题目 045：学习曲线_基础

要求：完成“学习曲线”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 在不同训练规模交叉验证。
3. 计算每个规模训练均值。
4. 计算每个规模验证均值。

完成标准：
- 验证曲线shape和范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.tree import DecisionTreeClassifier  # 导入对小样本规模稳定的模型。
from sklearn.model_selection import learning_curve  # 导入学习曲线工具。
features, labels = load_iris(return_X_y=True)  # 加载数据。
sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(max_depth=3, random_state=42), features, labels, train_sizes=[0.5, 0.75, 1.0], cv=3, scoring='accuracy', shuffle=True, random_state=42)  # 在不同训练规模交叉验证。
train_mean = train_scores.mean(axis=1)  # 计算每个规模训练均值。
valid_mean = valid_scores.mean(axis=1)  # 计算每个规模验证均值。
assert len(sizes) == 3 and np.all((0 <= valid_mean) & (valid_mean <= 1))  # 验证曲线shape和范围。
print(sizes, train_mean, valid_mean)  # 输出学习曲线数据。
