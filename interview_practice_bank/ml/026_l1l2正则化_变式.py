"""
题目 026：L1L2正则化_变式

要求：完成“L1L2正则化”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Elastic Net组合L1稀疏与L2稳定。
3. 验证组合正则项。

完成标准：
- 验证组合正则项。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
elastic_weights = np.array([-2.0, 0.5, 3.0]); l1_strength, l2_strength = 0.2, 0.3; elastic_penalty = l1_strength * np.abs(elastic_weights).sum() + l2_strength * (elastic_weights**2).sum()  # Elastic Net组合L1稀疏与L2稳定。
assert elastic_penalty > 0  # 验证组合正则项。
