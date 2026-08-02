"""
题目 030：偏差方差与过拟合_变式

要求：完成“偏差方差与过拟合”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Bootstrap重采样估计统计量方差。
3. 验证抽样分布。

完成标准：
- 验证抽样分布。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng_bias = np.random.default_rng(42); population = rng_bias.normal(size=200); bootstrap_means = np.array([rng_bias.choice(population, size=len(population), replace=True).mean() for _ in range(200)])  # Bootstrap重采样估计统计量方差。
assert bootstrap_means.std() > 0 and abs(bootstrap_means.mean() - population.mean()) < 0.1  # 验证抽样分布。
