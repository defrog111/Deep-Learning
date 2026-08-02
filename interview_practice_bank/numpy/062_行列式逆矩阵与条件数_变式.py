"""
题目 062：行列式逆矩阵与条件数_变式

要求：完成“行列式逆矩阵与条件数”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建可逆方阵。
3. slogdet在行列式极大或极小时更稳定。
4. 验证与det关系。

完成标准：
- 验证与det关系。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[4.0, 2.0], [1.0, 3.0]])  # 创建可逆方阵。
sign, log_abs_det = np.linalg.slogdet(matrix)  # slogdet在行列式极大或极小时更稳定。
assert np.isclose(sign * np.exp(log_abs_det), np.linalg.det(matrix))  # 验证与det关系。
