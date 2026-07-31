"""
题目 027：DecisionTree与过拟合控制_基础

要求：完成“DecisionTree与过拟合控制”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.model_selection import train_test_split  # 导入切分函数。
from sklearn.tree import DecisionTreeClassifier, export_text  # 导入决策树和文本导出。
features, labels = load_iris(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, stratify=labels, random_state=42)  # 分层切分。
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42).fit(train_x, train_y)  # 限制树深防止过拟合。
train_score, test_score = model.score(train_x, train_y), model.score(test_x, test_y)  # 对比训练和测试准确率。
rules = export_text(model, max_depth=2)  # 导出浅层规则便于解释。
assert model.get_depth() <= 3  # 验证复杂度约束。
print(train_score, test_score, '\n', rules)  # 输出性能和规则。
