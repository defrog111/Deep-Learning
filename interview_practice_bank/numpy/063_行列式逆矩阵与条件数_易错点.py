"""
题目 063：行列式逆矩阵与条件数_易错点

要求：完成“行列式逆矩阵与条件数”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建可逆方阵。
3. 求解Ax=I可得到逆，但实际预测应直接solve。
4. 验证两种结果。

完成标准：
- 验证两种结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[4.0, 2.0], [1.0, 3.0]])  # 创建可逆方阵。
identity_via_solve = np.linalg.solve(matrix, np.eye(matrix.shape[0]))  # 求解Ax=I可得到逆，但实际预测应直接solve。
assert np.allclose(identity_via_solve, np.linalg.inv(matrix))  # 验证两种结果。
