"""
题目 092：数据漂移PSI_综合

要求：完成“数据漂移PSI”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Jensen-Shannon是对称且有界的分布漂移指标。
3. 综合验证对称性。

完成标准：
- 综合验证对称性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
p_distribution = np.array([0.8, 0.2]); q_distribution = np.array([0.5, 0.5]); midpoint = (p_distribution + q_distribution) / 2; kl = lambda left, right: np.sum(left * np.log(left / right)); js_divergence = 0.5 * kl(p_distribution, midpoint) + 0.5 * kl(q_distribution, midpoint)  # Jensen-Shannon是对称且有界的分布漂移指标。
assert js_divergence >= 0 and np.isclose(js_divergence, 0.5 * kl(q_distribution, midpoint) + 0.5 * kl(p_distribution, midpoint))  # 综合验证对称性。
