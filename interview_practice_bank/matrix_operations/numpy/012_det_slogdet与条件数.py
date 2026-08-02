"""
题目 012：NumPy det_slogdet与条件数

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算行列式及稳定的对数行列式。
2. 比较良态与病态矩阵的条件数。
3. 解释条件数大时解对扰动敏感。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
well_conditioned = np.array([[2.0, 0.0], [0.0, 3.0]])  # 创建良态对角矩阵。
ill_conditioned = np.array([[1.0, 1.0], [1.0, 1.000001]])  # 创建两行近似相关的病态矩阵。
determinant = np.linalg.det(well_conditioned)  # 计算普通行列式。
sign, log_abs_det = np.linalg.slogdet(well_conditioned)  # 避免极大极小行列式上下溢。
good_cond = np.linalg.cond(well_conditioned)  # 计算良态矩阵条件数。
bad_cond = np.linalg.cond(ill_conditioned)  # 计算病态矩阵条件数。
assert np.isclose(sign * np.exp(log_abs_det), determinant) and bad_cond > good_cond * 1000  # 核对稳定表示和敏感性。
print(determinant, log_abs_det, good_cond, bad_cond)  # 输出数值稳定性指标。
