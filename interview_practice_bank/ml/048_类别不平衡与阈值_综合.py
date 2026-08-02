"""
题目 048：类别不平衡与阈值_综合

要求：完成“类别不平衡与阈值”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. balanced权重与类别频次成反比。
3. 综合验证加权守恒。

完成标准：
- 综合验证加权守恒。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
class_counts = np.array([90, 10]); balanced_weights = class_counts.sum() / (len(class_counts) * class_counts); sample_labels = np.array([0, 0, 1]); sample_weights = balanced_weights[sample_labels]  # balanced权重与类别频次成反比。
assert sample_weights[-1] > sample_weights[0] and np.isclose((balanced_weights * class_counts).sum(), class_counts.sum())  # 综合验证加权守恒。
