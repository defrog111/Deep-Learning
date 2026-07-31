"""
题目 086：多项式拟合_变式

要求：完成“多项式拟合”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建自变量采样点。
2. 根据二次函数生成无噪声目标。
3. 最小二乘拟合二次多项式。
4. 用拟合系数计算预测。
5. 无噪声且阶数正确时应精确拟合。
6. 新式Polynomial API可管理定义域并转换为普通系数。

完成标准：
- 无噪声且阶数正确时应精确拟合。
- 验证与polyfit预测一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
x = np.linspace(-2, 2, 9)  # 创建自变量采样点。
y = 2 * x**2 - 3 * x + 1  # 根据二次函数生成无噪声目标。
coefficients = np.polyfit(x, y, deg=2)  # 最小二乘拟合二次多项式。
predictions = np.polyval(coefficients, x)  # 用拟合系数计算预测。
assert np.allclose(predictions, y)  # 无噪声且阶数正确时应精确拟合。
print(coefficients, np.mean((predictions - y) ** 2))  # 输出系数和MSE。
polynomial = np.polynomial.Polynomial.fit(x, y, deg=2).convert()  # 新式Polynomial API可管理定义域并转换为普通系数。
assert np.allclose(polynomial(x), predictions)  # 验证与polyfit预测一致。
