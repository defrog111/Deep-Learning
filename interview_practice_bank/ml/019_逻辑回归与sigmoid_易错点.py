"""
题目 019：逻辑回归与Sigmoid_易错点

要求：完成“逻辑回归与Sigmoid”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 直接按logits计算BCE避免sigmoid上溢下溢。
3. 验证数值稳定性。

完成标准：
- 验证数值稳定性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
extreme_logits = np.array([-1000.0, 1000.0]); stable_losses = np.maximum(extreme_logits, 0) - extreme_logits * np.array([0.0, 1.0]) + np.log1p(np.exp(-np.abs(extreme_logits)))  # 直接按logits计算BCE避免sigmoid上溢下溢。
assert np.isfinite(stable_losses).all() and np.allclose(stable_losses, 0)  # 验证数值稳定性。
