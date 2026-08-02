"""
题目 020：RandomizedSearchCV_变式

要求：完成“RandomizedSearchCV”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 随机搜索可混合连续分布与离散候选。

完成标准：
- 验证抽样次数和参数范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from scipy.stats import loguniform  # 导入对数均匀分布。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.model_selection import RandomizedSearchCV  # 导入随机搜索。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # 导入标准化。
features, labels = load_iris(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from scipy.stats import randint  # 导入离散超参数分布。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林。
forest_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), {'n_estimators': randint(10, 40), 'max_depth': [None, 2, 4]}, n_iter=4, cv=3, random_state=42, n_jobs=1).fit(features, labels)  # 随机搜索可混合连续分布与离散候选。
assert len(forest_search.cv_results_['params']) == 4 and forest_search.best_params_['n_estimators'] >= 10  # 验证抽样次数和参数范围。
