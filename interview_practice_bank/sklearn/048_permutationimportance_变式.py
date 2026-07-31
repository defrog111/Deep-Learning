"""
题目 048：PermutationImportance_变式

要求：完成“PermutationImportance”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 创建独立测试集。
3. 训练模型。
4. 在测试集打乱单列测性能下降。
5. 按平均下降排序。

完成标准：
- 验证每个特征都有结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.inspection import permutation_importance  # 导入排列重要性。
from sklearn.model_selection import train_test_split  # 导入切分函数。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林。
features, labels = load_iris(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, stratify=labels, random_state=42)  # 创建独立测试集。
model = RandomForestClassifier(n_estimators=40, random_state=42).fit(train_x, train_y)  # 训练模型。
importance = permutation_importance(model, test_x, test_y, n_repeats=5, random_state=42)  # 在测试集打乱单列测性能下降。
order = np.argsort(importance.importances_mean)[::-1]  # 按平均下降排序。
assert len(order) == features.shape[1]  # 验证每个特征都有结果。
print(order, importance.importances_mean)  # 输出排列重要性。
