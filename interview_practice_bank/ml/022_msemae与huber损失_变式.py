"""
题目 022：MSEMAE与Huber损失_变式

要求：完成“MSEMAE与Huber损失”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Pinball损失用于分位数回归且对正负误差不对称。
3. 验证高分位更惩罚低估。

完成标准：
- 验证高分位更惩罚低估。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
quantile_errors = np.array([-2.0, -1.0, 1.0, 3.0]); quantile = 0.8; pinball = np.maximum(quantile * quantile_errors, (quantile - 1) * quantile_errors)  # Pinball损失用于分位数回归且对正负误差不对称。
assert pinball.tolist() == [0.3999999999999999, 0.19999999999999996, 0.8, 2.4000000000000004]  # 验证高分位更惩罚低估。
