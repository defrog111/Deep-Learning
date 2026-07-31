"""
题目 011：Pipeline防止数据泄漏_基础

要求：完成“Pipeline防止数据泄漏”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 把预处理放进每折内部防泄漏。
3. 对完整Pipeline交叉验证。

完成标准：
- 验证模型表现合理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
