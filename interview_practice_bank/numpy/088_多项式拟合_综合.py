"""
题目 088：多项式拟合_综合

要求：完成“多项式拟合”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建自变量采样点。
2. 根据二次函数生成无噪声目标。
3. 最小二乘拟合二次多项式。
4. 用拟合系数计算预测。
5. 无噪声且阶数正确时应精确拟合。
6. 综合用Vandermonde设计矩阵理解多项式拟合。

完成标准：
- 无噪声且阶数正确时应精确拟合。
- 验证底层最小二乘形式。
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
vandermonde = np.vander(x, N=3); manual_coefficients = np.linalg.lstsq(vandermonde, y, rcond=None)[0]  # 综合用Vandermonde设计矩阵理解多项式拟合。
assert np.allclose(vandermonde @ manual_coefficients, predictions)  # 验证底层最小二乘形式。
