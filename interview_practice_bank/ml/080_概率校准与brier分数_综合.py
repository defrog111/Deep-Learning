"""
题目 080：概率校准与Brier分数_综合

要求：完成“概率校准与Brier分数”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Platt scaling在独立校准集上拟合一维逻辑映射。
3. 综合验证单调概率映射。

完成标准：
- 综合验证单调概率映射。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
raw_calibration_scores = np.array([-2.0, -1.0, 1.0, 2.0]); platt_a, platt_b = 1.2, -0.1; platt_probabilities = 1 / (1 + np.exp(-(platt_a * raw_calibration_scores + platt_b)))  # Platt scaling在独立校准集上拟合一维逻辑映射。
assert np.all(np.diff(platt_probabilities) > 0) and np.all((platt_probabilities > 0) & (platt_probabilities < 1))  # 综合验证单调概率映射。
