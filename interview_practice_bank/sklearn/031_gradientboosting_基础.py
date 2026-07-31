"""
题目 031：GradientBoosting_基础

要求：完成“GradientBoosting”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升树。
from sklearn.model_selection import train_test_split  # 导入数据切分。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, stratify=labels, random_state=42)  # 分层切分。
model = GradientBoostingClassifier(n_estimators=40, learning_rate=0.05, max_depth=2, random_state=42).fit(train_x, train_y)  # 小学习率逐步拟合残差。
staged_scores = [(stage_predictions == test_y).mean() for stage_predictions in model.staged_predict(test_x)]  # 观察每一阶段测试准确率。
assert len(staged_scores) == model.n_estimators  # 验证每棵树对应一个阶段。
print(model.score(test_x, test_y), max(staged_scores))  # 输出最终和最佳阶段分数。
