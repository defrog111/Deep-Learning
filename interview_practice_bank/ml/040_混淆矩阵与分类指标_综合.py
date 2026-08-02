"""
题目 040：混淆矩阵与分类指标_综合

要求：完成“混淆矩阵与分类指标”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合计算多标签micro-F1。
3. 验证样本类别展开后的整体指标。

完成标准：
- 验证样本类别展开后的整体指标。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
multilabel_true = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool); multilabel_pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool); micro_tp = (multilabel_true & multilabel_pred).sum(); micro_fp = (~multilabel_true & multilabel_pred).sum(); micro_fn = (multilabel_true & ~multilabel_pred).sum(); micro_f1 = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)  # 综合计算多标签micro-F1。
assert np.isclose(micro_f1, 2 / 3)  # 验证样本类别展开后的整体指标。
