"""
题目 018：GridSearchCV参数搜索_变式

要求：完成“GridSearchCV参数搜索”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
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
