"""
题目 030：偏差方差与过拟合_变式

要求：完成“偏差方差与过拟合”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 模型复杂度增大时训练误差下降。
2. 用验证集选择复杂度。
3. 计算泛化差距。
4. 判断复杂模型是否高方差。

完成标准：
- 验证选择验证误差最低模型。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
train_errors = np.array([0.40, 0.25, 0.12, 0.05])  # 模型复杂度增大时训练误差下降。
validation_errors = np.array([0.45, 0.28, 0.18, 0.30])  # 验证误差先降后升。
best_complexity = validation_errors.argmin() + 1  # 用验证集选择复杂度。
generalization_gap = validation_errors - train_errors  # 计算泛化差距。
high_variance = generalization_gap[-1] > 0.08  # 判断复杂模型是否高方差。
assert best_complexity == 3  # 验证选择验证误差最低模型。
print(best_complexity, generalization_gap, high_variance)  # 输出偏差方差线索。
