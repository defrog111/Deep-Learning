"""
题目 030：RandomForest特征重要性_变式

要求：完成“RandomForest特征重要性”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. ExtraTrees随机阈值降低相关性但增大单树偏差。

完成标准：
- 验证集成预测和重要性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林。
features, labels = load_iris(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.ensemble import ExtraTreesClassifier  # 导入更随机的极端随机树。
extra_trees = ExtraTreesClassifier(n_estimators=40, max_features='sqrt', random_state=42).fit(features, labels); impurity_importance = extra_trees.feature_importances_  # ExtraTrees随机阈值降低相关性但增大单树偏差。
assert np.isclose(impurity_importance.sum(), 1) and extra_trees.predict(features[:3]).shape == (3,)  # 验证集成预测和重要性。
