"""
题目 023：MSEMAE与Huber损失_易错点

要求：完成“MSEMAE与Huber损失”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. MSE平方放大离群点，MAE线性增长。
3. 验证鲁棒性差异。

完成标准：
- 验证鲁棒性差异。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
outlier_errors = np.array([0.0, 1.0, 100.0]); mse_contribution = outlier_errors**2; mae_contribution = np.abs(outlier_errors)  # MSE平方放大离群点，MAE线性增长。
assert mse_contribution[-1] / mse_contribution[1] > mae_contribution[-1] / mae_contribution[1]  # 验证鲁棒性差异。
