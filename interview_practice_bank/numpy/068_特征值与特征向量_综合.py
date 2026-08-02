"""
题目 068：特征值与特征向量_综合

要求：完成“特征值与特征向量”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建实对称矩阵。
3. 用一次幂迭代逼近主特征向量方向。
4. 验证归一化。

完成标准：
- 验证归一化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
power_vector = np.ones(matrix.shape[0]); power_vector = matrix @ power_vector; power_vector /= np.linalg.norm(power_vector)  # 用一次幂迭代逼近主特征向量方向。
assert np.isclose(np.linalg.norm(power_vector), 1)  # 验证归一化。
