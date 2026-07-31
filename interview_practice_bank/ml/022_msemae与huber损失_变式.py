"""
题目 022：MSEMAE与Huber损失_变式

要求：完成“MSEMAE与Huber损失”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造含离群点目标。
2. 构造预测。
3. 计算残差。
4. MSE对大误差平方惩罚。
5. MAE对离群点更稳健。
6. 设置Huber转折点。
7. 分段计算Huber损失。
8. 本例离群点使MSE明显更大。
9. Pinball损失用于分位数回归且对正负误差不对称。

完成标准：
- 本例离群点使MSE明显更大。
- 验证高分位更惩罚低估。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
targets = np.array([1.0, 2.0, 3.0, 20.0])  # 构造含离群点目标。
predictions = np.array([1.2, 1.8, 3.1, 5.0])  # 构造预测。
errors = predictions - targets  # 计算残差。
mse = np.mean(errors**2)  # MSE对大误差平方惩罚。
mae = np.mean(np.abs(errors))  # MAE对离群点更稳健。
delta = 4.0  # 设置Huber转折点。
huber = np.mean(np.where(np.abs(errors) <= delta, 0.5 * errors**2, delta * (np.abs(errors) - 0.5 * delta)))  # 分段计算Huber损失。
assert mse > mae  # 本例离群点使MSE明显更大。
print(mse, mae, huber)  # 输出三种回归损失。
quantile_errors = np.array([-2.0, -1.0, 1.0, 3.0]); quantile = 0.8; pinball = np.maximum(quantile * quantile_errors, (quantile - 1) * quantile_errors)  # Pinball损失用于分位数回归且对正负误差不对称。
assert pinball.tolist() == [0.3999999999999999, 0.19999999999999996, 0.8, 2.4000000000000004]  # 验证高分位更惩罚低估。
