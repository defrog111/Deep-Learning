"""
题目 086：多项式拟合_变式

要求：完成“多项式拟合”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 用独立数据学习新式Polynomial API和系数升幂顺序。
3. 验证预测和新式系数顺序。

完成标准：
- 验证预测和新式系数顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
sample_x = np.linspace(0, 4, 9); sample_y = 0.5 * sample_x**2 + 2; polynomial = np.polynomial.Polynomial.fit(sample_x, sample_y, deg=2).convert()  # 用独立数据学习新式Polynomial API和系数升幂顺序。
assert np.allclose(polynomial(sample_x), sample_y) and np.allclose(polynomial.coef, [2, 0, 0.5])  # 验证预测和新式系数顺序。
