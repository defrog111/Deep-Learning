"""
题目 035：KMeans与silhouette_基础

要求：完成“KMeans与silhouette”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入NumPy以局部管理当前平台的矩阵运算警告。
from sklearn.cluster import KMeans  # 导入KMeans。
from sklearn.datasets import make_blobs  # 导入聚类数据生成器。
from sklearn.metrics import silhouette_score  # 导入轮廓系数。
features, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.6, random_state=42)  # 生成三个簇。
model = KMeans(n_clusters=3, n_init=10, algorithm='elkan', random_state=42)  # 使用多次初始化和Elkan算法降低局部最优风险。
with np.errstate(over='ignore', invalid='ignore', divide='ignore'):  # 仅屏蔽部分macOS Accelerate与NumPy组合的已知伪警告。
    model.fit(features)  # 在局部数值错误策略下拟合KMeans。
    score = silhouette_score(features, model.labels_)  # 计算簇内紧密簇间分离指标。
assert model.cluster_centers_.shape[0] == 3  # 验证中心数量。
print(model.inertia_, score, model.cluster_centers_)  # 输出惯性、轮廓系数和中心。
