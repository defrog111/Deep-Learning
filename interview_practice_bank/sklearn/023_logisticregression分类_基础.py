"""
题目 023：LogisticRegression分类_基础

要求：完成“LogisticRegression分类”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 分层切分。
3. 训练多分类逻辑回归。
4. 概率最大类别即预测。

完成标准：
- 验证准确率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.metrics import accuracy_score, log_loss  # 导入分类指标。
from sklearn.model_selection import train_test_split  # 导入切分函数。
features, labels = load_iris(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=41, stratify=labels)  # 分层切分。
model = LogisticRegression(max_iter=500).fit(train_x, train_y)  # 训练多分类逻辑回归。
probabilities = model.predict_proba(test_x)  # 输出每类概率。
predictions = probabilities.argmax(axis=1)  # 概率最大类别即预测。
assert accuracy_score(test_y, predictions) > 0.85  # 验证准确率。
print(accuracy_score(test_y, predictions), log_loss(test_y, probabilities))  # 输出硬指标和概率指标。
