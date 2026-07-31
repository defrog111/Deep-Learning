"""
题目 018：GridSearchCV参数搜索_变式

要求：完成“GridSearchCV参数搜索”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 建立预处理加模型流水线。
3. 使用步骤名双下划线搜索参数。
4. 综合题统一导入NumPy用于shape、数值和标签检查。
5. 多指标搜索必须指定refit依据。

完成标准：
- 验证找到最优参数。
- 验证搜索结果结构。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.model_selection import GridSearchCV  # 导入网格搜索。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
from sklearn.svm import SVC  # 导入支持向量机。
features, labels = load_iris(return_X_y=True)  # 加载数据。
pipeline = make_pipeline(StandardScaler(), SVC())  # 建立预处理加模型流水线。
search = GridSearchCV(pipeline, {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}, cv=3, scoring='accuracy').fit(features, labels)  # 使用步骤名双下划线搜索参数。
assert 'svc__C' in search.best_params_  # 验证找到最优参数。
print(search.best_params_, search.best_score_)  # 输出最优配置。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # 导入网格搜索和显式分层内层CV。
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42); multi_search = GridSearchCV(pipeline, {'svc__C': [0.1, 1, 10]}, scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}, refit='f1_macro', cv=inner_cv, return_train_score=True).fit(features, labels)  # 多指标搜索必须指定refit依据。
assert multi_search.best_estimator_ is not None and 'mean_train_accuracy' in multi_search.cv_results_  # 验证搜索结果结构。
