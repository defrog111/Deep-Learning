"""
题目 062：行列式逆矩阵与条件数_变式

要求：完成“行列式逆矩阵与条件数”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建可逆方阵。
2. 计算行列式判断是否奇异。
3. 计算逆矩阵用于教学展示。
4. 原矩阵乘逆矩阵应得到单位阵。
5. 条件数衡量数值敏感性。

完成标准：
- 验证可逆性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[4.0, 2.0], [1.0, 3.0]])  # 创建可逆方阵。
determinant = np.linalg.det(matrix)  # 计算行列式判断是否奇异。
inverse = np.linalg.inv(matrix)  # 计算逆矩阵用于教学展示。
identity = matrix @ inverse  # 原矩阵乘逆矩阵应得到单位阵。
condition = np.linalg.cond(matrix)  # 条件数衡量数值敏感性。
assert determinant != 0 and np.allclose(identity, np.eye(2))  # 验证可逆性。
print(determinant, inverse, condition)  # 输出线性代数指标。
