"""
题目 027：L1L2正则化_易错点

要求：完成“L1L2正则化”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 未标准化时同等预测作用的特征受到完全不同L1惩罚。
3. 验证正则化前必须关注尺度。

完成标准：
- 验证正则化前必须关注尺度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
scaled_equivalent = np.array([1.0, 1000.0]); same_effect_weights = np.array([1.0, 0.001]); unequal_penalty = np.abs(same_effect_weights)  # 未标准化时同等预测作用的特征受到完全不同L1惩罚。
assert unequal_penalty[0] / unequal_penalty[1] == 1000  # 验证正则化前必须关注尺度。
