"""
题目 078：概率校准与Brier分数_变式

要求：完成“概率校准与Brier分数”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. ECE按bin加权置信度与准确率差。
3. 验证校准误差范围。

完成标准：
- 验证校准误差范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
calibration_probabilities = np.array([0.1, 0.2, 0.8, 0.9]); calibration_labels = np.array([0, 1, 1, 1]); calibration_bins = np.array([0, 0, 1, 1]); bin_gaps = [abs(calibration_probabilities[calibration_bins == index].mean() - calibration_labels[calibration_bins == index].mean()) for index in range(2)]; ece = sum((calibration_bins == index).mean() * bin_gaps[index] for index in range(2))  # ECE按bin加权置信度与准确率差。
assert 0 <= ece <= 1  # 验证校准误差范围。
