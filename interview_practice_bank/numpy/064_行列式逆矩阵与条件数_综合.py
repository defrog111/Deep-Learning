"""
题目 064：行列式逆矩阵与条件数_综合

要求：完成“行列式逆矩阵与条件数”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建可逆方阵。
3. 综合使用正则化和伪逆处理病态矩阵。
4. 验证数值结果有限。

完成标准：
- 验证数值结果有限。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[4.0, 2.0], [1.0, 3.0]])  # 创建可逆方阵。
regularized = matrix.T @ matrix + 1e-6 * np.eye(matrix.shape[1]); stable_inverse = np.linalg.pinv(regularized)  # 综合使用正则化和伪逆处理病态矩阵。
assert np.isfinite(stable_inverse).all()  # 验证数值结果有限。
