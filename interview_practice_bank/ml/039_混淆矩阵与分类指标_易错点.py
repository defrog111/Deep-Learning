"""
题目 039：混淆矩阵与分类指标_易错点

要求：完成“混淆矩阵与分类指标”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. specificity关注负类，balanced accuracy平均两类召回。
3. 验证不平衡指标。

完成标准：
- 验证不平衡指标。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
binary_confusion = np.array([[90, 10], [5, 15]]); tn, fp, fn, tp = binary_confusion.ravel(); specificity = tn / (tn + fp); balanced_accuracy = (tp / (tp + fn) + specificity) / 2  # specificity关注负类，balanced accuracy平均两类召回。
assert np.isclose(specificity, 0.9) and np.isclose(balanced_accuracy, 0.825)  # 验证不平衡指标。
