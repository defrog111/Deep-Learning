"""
题目 010：NumPy lstsq最小二乘回归

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 为线性模型增加截距列。
2. 用最小二乘拟合含噪数据。
3. 检查残差、秩和奇异值。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
x = np.arange(5.0)  # 创建单特征样本。
design = np.column_stack((x, np.ones_like(x)))  # 添加全1列以同时拟合斜率和截距。
targets = np.array([1.1, 2.9, 5.2, 6.8, 9.1])  # 创建近似 y=2x+1 的含噪目标。
parameters, residuals, rank, singular_values = np.linalg.lstsq(design, targets, rcond=None)  # 求最小二乘解。
predictions = design @ parameters  # 用矩阵乘法计算拟合值。
assert rank == 2 and parameters.shape == (2,) and residuals.size == 1  # 检查满列秩和返回结构。
print(parameters, predictions, residuals, singular_values)  # 输出拟合诊断信息。
