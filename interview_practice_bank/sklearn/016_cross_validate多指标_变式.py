"""
题目 016：cross_validate多指标_变式

要求：完成“cross_validate多指标”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 建立可复现的分类模型。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. F2更重视召回并返回每折模型。

完成标准：
- 验证自定义指标和估计器集合。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入非线性分类器。
from sklearn.model_selection import cross_validate  # 导入多指标交叉验证。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
model = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)  # 建立可复现的分类模型。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.metrics import make_scorer, fbeta_score  # 导入自定义评分器。
from sklearn.model_selection import cross_validate  # 导入多指标验证。
custom_scorer = make_scorer(fbeta_score, beta=2, average='macro'); custom_result = cross_validate(model, features, labels, cv=3, scoring={'f2_macro': custom_scorer, 'accuracy': 'accuracy'}, return_estimator=True)  # F2更重视召回并返回每折模型。
assert len(custom_result['estimator']) == 3 and 'test_f2_macro' in custom_result  # 验证自定义指标和估计器集合。
