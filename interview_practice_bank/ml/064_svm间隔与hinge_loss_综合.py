"""
题目 064：SVM间隔与hinge loss_综合

要求：完成“SVM间隔与hinge loss”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 合法Mercer核矩阵应半正定。
3. 综合核SVM决策形式。

完成标准：
- 综合核SVM决策形式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
kernel_matrix = np.array([[1.0, 0.5], [0.5, 1.0]]); kernel_eigenvalues = np.linalg.eigvalsh(kernel_matrix); decision_scores = kernel_matrix @ np.array([1.0, -1.0]) + 0.1  # 合法Mercer核矩阵应半正定。
assert kernel_eigenvalues.min() >= 0 and decision_scores.shape == (2,)  # 综合核SVM决策形式。
