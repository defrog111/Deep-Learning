"""
题目 019：RandomizedSearchCV_基础

要求：完成“RandomizedSearchCV”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from scipy.stats import loguniform  # 导入对数均匀分布。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.model_selection import RandomizedSearchCV  # 导入随机搜索。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # 导入标准化。
features, labels = load_iris(return_X_y=True)  # 加载数据。
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))  # 建立Pipeline。
search = RandomizedSearchCV(pipeline, {'logisticregression__C': loguniform(1e-3, 1e2)}, n_iter=4, cv=3, random_state=42).fit(features, labels)  # 从连续分布随机抽取超参数。
assert search.best_estimator_ is not None  # 验证搜索完成。
print(search.best_params_, search.best_score_)  # 输出最优参数和分数。
