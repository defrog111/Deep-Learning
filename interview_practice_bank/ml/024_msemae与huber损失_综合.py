"""
题目 024：MSEMAE与Huber损失_综合

要求：完成“MSEMAE与Huber损失”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. log-cosh在零附近像MSE、大误差时像MAE。
3. 验证平滑对称损失。

完成标准：
- 验证平滑对称损失。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
log_cosh_errors = np.array([-10.0, 0.0, 10.0]); log_cosh = np.logaddexp(log_cosh_errors, -log_cosh_errors) - np.log(2)  # log-cosh在零附近像MSE、大误差时像MAE。
assert np.isfinite(log_cosh).all() and log_cosh[0] == log_cosh[-1]  # 验证平滑对称损失。
