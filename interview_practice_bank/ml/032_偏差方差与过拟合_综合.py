"""
题目 032：偏差方差与过拟合_综合

要求：完成“偏差方差与过拟合”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 平均低相关模型可降低方差。
3. 综合解释bagging改善高方差。

完成标准：
- 综合解释bagging改善高方差。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
ensemble_errors = np.array([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [0.5, -0.5, 0.0]]); individual_variance = ensemble_errors.var(axis=1).mean(); averaged_variance = ensemble_errors.mean(axis=0).var()  # 平均低相关模型可降低方差。
assert averaged_variance < individual_variance  # 综合解释bagging改善高方差。
