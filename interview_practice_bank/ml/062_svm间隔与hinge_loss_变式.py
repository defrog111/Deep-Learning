"""
题目 062：SVM间隔与hinge loss_变式

要求：完成“SVM间隔与hinge loss”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 间隔内或边界上的样本影响hinge loss。
3. 验证支持向量候选。

完成标准：
- 验证支持向量候选。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
svm_scores = np.array([2.0, 0.5, -1.5]); svm_labels = np.array([1.0, 1.0, -1.0]); margins = svm_labels * svm_scores; support_candidates = margins <= 1  # 间隔内或边界上的样本影响hinge loss。
assert support_candidates.tolist() == [False, True, False]  # 验证支持向量候选。
