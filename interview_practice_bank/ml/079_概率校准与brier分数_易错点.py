"""
题目 079：概率校准与Brier分数_易错点

要求：完成“概率校准与Brier分数”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Brier分数会严惩高置信度错误。
3. 验证概率质量不等于准确率。

完成标准：
- 验证概率质量不等于准确率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
overconfident = np.array([0.99, 0.99]); uncertain = np.array([0.6, 0.6]); wrong_labels = np.array([1.0, 0.0]); overconfident_brier = np.mean((overconfident - wrong_labels)**2); uncertain_brier = np.mean((uncertain - wrong_labels)**2)  # Brier分数会严惩高置信度错误。
assert overconfident_brier > uncertain_brier  # 验证概率质量不等于准确率。
