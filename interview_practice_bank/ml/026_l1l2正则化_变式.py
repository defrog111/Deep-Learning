"""
题目 026：L1L2正则化_变式

要求：完成“L1L2正则化”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
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
