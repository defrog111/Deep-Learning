"""
题目 068：KMeans聚类_综合

要求：完成“KMeans聚类”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建两个明显簇。
2. 初始化两个中心。
3. 交替执行分配和更新。
4. 计算样本到各中心平方距离。
5. 分配到最近中心。
6. 更新为簇均值。
7. 计算簇内平方和。
8. 综合演示GMM的E步责任度与M步均值更新。

完成标准：
- 验证简单数据被正确聚类。
- 验证软聚类。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
points = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])  # 创建两个明显簇。
centers = np.array([[0.0, 0.0], [5.0, 5.0]])  # 初始化两个中心。
for _ in range(5):  # 交替执行分配和更新。
    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # 计算样本到各中心平方距离。
    assignments = distances.argmin(axis=1)  # 分配到最近中心。
    centers = np.stack([points[assignments == cluster].mean(axis=0) for cluster in range(2)])  # 更新为簇均值。
inertia = np.sum((points - centers[assignments]) ** 2)  # 计算簇内平方和。
assert inertia <= 1.0  # 验证简单数据被正确聚类。
print(assignments, centers, inertia)  # 输出聚类结果。
gmm_values = np.array([-2.0, -1.5, 1.5, 2.0]); gmm_means = np.array([-1.0, 1.0]); responsibilities = np.exp(-0.5 * (gmm_values[:, None] - gmm_means[None, :])**2); responsibilities /= responsibilities.sum(1, keepdims=True); updated_means = (responsibilities * gmm_values[:, None]).sum(0) / responsibilities.sum(0)  # 综合演示GMM的E步责任度与M步均值更新。
assert updated_means[0] < 0 < updated_means[1] and np.allclose(responsibilities.sum(1), 1)  # 验证软聚类。
