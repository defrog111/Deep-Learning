"""
题目 016：梯度下降_综合

要求：完成“梯度下降”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 初始化Adam的一阶二阶矩。
for step in range(1, 101):  # 迭代最小化(value-3)^2。
    gradient = 2 * (adam_value - 3); first_moment = 0.9 * first_moment + 0.1 * gradient; second_moment = 0.999 * second_moment + 0.001 * gradient**2; corrected_first = first_moment / (1 - 0.9**step); corrected_second = second_moment / (1 - 0.999**step); adam_value -= 0.1 * corrected_first / (np.sqrt(corrected_second) + 1e-8)  # 偏差修正后更新。
3. 验证Adam综合实现。

完成标准：
- 验证Adam综合实现。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
adam_value = 0.0; first_moment = second_moment = 0.0  # 初始化Adam的一阶二阶矩。
for step in range(1, 101):  # 迭代最小化(value-3)^2。
    gradient = 2 * (adam_value - 3); first_moment = 0.9 * first_moment + 0.1 * gradient; second_moment = 0.999 * second_moment + 0.001 * gradient**2; corrected_first = first_moment / (1 - 0.9**step); corrected_second = second_moment / (1 - 0.999**step); adam_value -= 0.1 * corrected_first / (np.sqrt(corrected_second) + 1e-8)  # 偏差修正后更新。
assert abs(adam_value - 3) < 0.1  # 验证Adam综合实现。
