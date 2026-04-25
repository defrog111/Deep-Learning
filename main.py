# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: NumPy KMeans.\"\"\"  # 说明这道题是手写 KMeans。

import numpy as np  # 导入 NumPy，用来做距离计算和均值更新。

x = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0]], dtype=np.float32)  # 定义样本矩阵，shape = (6, 2)。
centroids = x[:2].copy()  # 用前两个样本初始化两个聚类中心，shape = (2, 2)。
k = centroids.shape[0]  # 记录聚类中心个数 k = 2。

for step in range(5):  # 迭代更新 5 次。
    distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)  # 计算距离矩阵，shape = (6, 2)。
    labels = np.argmin(distances, axis=1)  # 为每个样本分配最近中心编号，labels 的 shape = (6,)。
    new_centroids = np.zeros_like(centroids)  # 创建新中心矩阵，shape = (2, 2)。
    for cluster_id in range(k):  # 逐个更新每个簇中心。
        cluster_points = x[labels == cluster_id]  # 取出当前簇样本，shape = (num_points_in_cluster, 2)。
        new_centroids[cluster_id] = np.mean(cluster_points, axis=0)  # 对当前簇求均值作为新中心，shape = (2,)。
    centroids = new_centroids  # 用新中心替换旧中心。
    print(f\"Step {step} | Labels: {labels} | Centroids: {centroids}\")  # 打印当前分配和中心。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
inertia = np.sum((x - centroids[labels]) ** 2)  # è®¡ç®—ç°‡å†…å¹³æ–¹å’Œ Inertiaï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Inertia:", float(inertia))  # æ‰“å° KMeans å¸¸è§æŒ‡æ ‡ Inertiaã€‚
