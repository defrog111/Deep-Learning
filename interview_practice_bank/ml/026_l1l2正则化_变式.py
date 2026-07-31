"""
题目 026：L1L2正则化_变式

要求：完成“L1L2正则化”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造模型权重。
2. 设置正则化强度。
3. L1倾向产生稀疏权重。
4. L2平滑缩小大权重。
5. 计算非零处L1次梯度。
6. 计算L2梯度。
7. 正则项应非负。
8. Elastic Net组合L1稀疏与L2稳定。

完成标准：
- 正则项应非负。
- 验证组合正则项。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
weights = np.array([3.0, -1.0, 0.2, 0.0])  # 构造模型权重。
strength = 0.2  # 设置正则化强度。
l1_penalty = strength * np.abs(weights).sum()  # L1倾向产生稀疏权重。
l2_penalty = strength * (weights**2).sum()  # L2平滑缩小大权重。
l1_subgradient = strength * np.sign(weights)  # 计算非零处L1次梯度。
l2_gradient = 2 * strength * weights  # 计算L2梯度。
assert l1_penalty >= 0 and l2_penalty >= 0  # 正则项应非负。
print(l1_penalty, l2_penalty, l1_subgradient, l2_gradient)  # 输出惩罚和梯度。
elastic_weights = np.array([-2.0, 0.5, 3.0]); l1_strength, l2_strength = 0.2, 0.3; elastic_penalty = l1_strength * np.abs(elastic_weights).sum() + l2_strength * (elastic_weights**2).sum()  # Elastic Net组合L1稀疏与L2稳定。
assert elastic_penalty > 0  # 验证组合正则项。
