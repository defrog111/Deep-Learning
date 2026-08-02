"""
题目 038：混淆矩阵与分类指标_变式

要求：完成“混淆矩阵与分类指标”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 比较macro和按支持度weighted平均。
3. 验证多分类指标范围。

完成标准：
- 验证多分类指标范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
multi_confusion = np.array([[5, 1, 0], [2, 3, 1], [0, 1, 7]]); per_class_recall = np.diag(multi_confusion) / multi_confusion.sum(axis=1); macro_recall = per_class_recall.mean(); weighted_recall = np.average(per_class_recall, weights=multi_confusion.sum(axis=1))  # 比较macro和按支持度weighted平均。
assert 0 <= macro_recall <= 1 and 0 <= weighted_recall <= 1  # 验证多分类指标范围。
