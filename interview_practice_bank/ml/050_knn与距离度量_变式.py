"""
题目 050：KNN与距离度量_变式

要求：完成“KNN与距离度量”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. KNN对尺度敏感，缩放会改变距离贡献。
3. 验证尺度影响。

完成标准：
- 验证尺度影响。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
knn_train = np.array([[0.0, 0.0], [1.0, 1000.0]]); knn_query = np.array([0.9, 0.0]); raw_distances = np.linalg.norm(knn_train - knn_query, axis=1); standardized_train = (knn_train - knn_train.mean(0)) / np.where(knn_train.std(0) == 0, 1, knn_train.std(0)); standardized_query = (knn_query - knn_train.mean(0)) / np.where(knn_train.std(0) == 0, 1, knn_train.std(0))  # KNN对尺度敏感，缩放会改变距离贡献。
assert not np.allclose(raw_distances, np.linalg.norm(standardized_train - standardized_query, axis=1))  # 验证尺度影响。
