"""
题目 068：KMeans聚类_综合

要求：完成“KMeans聚类”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合演示GMM的E步责任度与M步均值更新。
3. 验证软聚类。

完成标准：
- 验证软聚类。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
gmm_values = np.array([-2.0, -1.5, 1.5, 2.0]); gmm_means = np.array([-1.0, 1.0]); responsibilities = np.exp(-0.5 * (gmm_values[:, None] - gmm_means[None, :])**2); responsibilities /= responsibilities.sum(1, keepdims=True); updated_means = (responsibilities * gmm_values[:, None]).sum(0) / responsibilities.sum(0)  # 综合演示GMM的E步责任度与M步均值更新。
assert updated_means[0] < 0 < updated_means[1] and np.allclose(responsibilities.sum(1), 1)  # 验证软聚类。
