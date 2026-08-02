"""
题目 067：KMeans聚类_易错点

要求：完成“KMeans聚类”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. DBSCAN用eps邻域和min_samples识别密度及噪声。
3. 验证离群点。

完成标准：
- 验证离群点。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
dbscan_points = np.array([[0.0], [0.1], [5.0], [5.1], [20.0]]); dbscan_distances = np.abs(dbscan_points - dbscan_points.T); neighbor_counts = (dbscan_distances <= 0.2).sum(1); core_mask = neighbor_counts >= 2; noise_candidate = ~core_mask  # DBSCAN用eps邻域和min_samples识别密度及噪声。
assert noise_candidate.tolist() == [False, False, False, False, True]  # 验证离群点。
