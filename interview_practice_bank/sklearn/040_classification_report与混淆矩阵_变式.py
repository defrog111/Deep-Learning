"""
题目 040：classification_report与混淆矩阵_变式

要求：完成“classification_report与混淆矩阵”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 每个预测都来自未见该样本的折。
3. 构建混淆矩阵。
4. 获取precision recall F1字典。

完成标准：
- 验证每个样本被统计一次。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入分类器。
from sklearn.metrics import classification_report, confusion_matrix  # 导入综合指标。
from sklearn.model_selection import cross_val_predict  # 导入无泄漏交叉验证预测。
features, labels = load_iris(return_X_y=True)  # 加载数据。
predictions = cross_val_predict(RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42), features, labels, cv=5)  # 每个预测都来自未见该样本的折。
matrix = confusion_matrix(labels, predictions)  # 构建混淆矩阵。
report = classification_report(labels, predictions, output_dict=True)  # 获取precision recall F1字典。
assert matrix.sum() == len(labels)  # 验证每个样本被统计一次。
print(matrix, report['macro avg'])  # 输出矩阵和宏平均。
