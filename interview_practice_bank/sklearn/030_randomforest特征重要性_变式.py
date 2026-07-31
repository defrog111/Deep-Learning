"""
题目 030：RandomForest特征重要性_变式

要求：完成“RandomForest特征重要性”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 训练带袋外评估的森林。
3. 取得基于不纯度的特征重要性。
4. 按重要性排序。
5. 综合题统一导入NumPy用于shape、数值和标签检查。
6. ExtraTrees随机阈值降低相关性但增大单树偏差。

完成标准：
- 验证重要性归一化和OOB表现。
- 验证集成预测和重要性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林。
features, labels = load_iris(return_X_y=True)  # 加载数据。
model = RandomForestClassifier(n_estimators=40, max_depth=4, oob_score=True, bootstrap=True, random_state=42).fit(features, labels)  # 训练带袋外评估的森林。
importance = model.feature_importances_  # 取得基于不纯度的特征重要性。
order = np.argsort(importance)[::-1]  # 按重要性排序。
assert np.isclose(importance.sum(), 1) and model.oob_score_ > 0.8  # 验证重要性归一化和OOB表现。
print(order, importance, model.oob_score_)  # 输出特征排名。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.ensemble import ExtraTreesClassifier  # 导入更随机的极端随机树。
extra_trees = ExtraTreesClassifier(n_estimators=40, max_features='sqrt', random_state=42).fit(features, labels); impurity_importance = extra_trees.feature_importances_  # ExtraTrees随机阈值降低相关性但增大单树偏差。
assert np.isclose(impurity_importance.sum(), 1) and extra_trees.predict(features[:3]).shape == (3,)  # 验证集成预测和重要性。
