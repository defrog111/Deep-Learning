"""
题目 087：多项式拟合_易错点

要求：完成“多项式拟合”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 从已知多项式独立计算根并代回。
3. 验证根和数值误差范围。

完成标准：
- 验证根和数值误差范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
known_coefficients = np.array([1.0, -3.0, 2.0]); roots = np.roots(known_coefficients); evaluated = np.polyval(known_coefficients, roots)  # 从已知多项式独立计算根并代回。
assert np.allclose(np.sort(roots), [1, 2]) and np.allclose(evaluated, 0, atol=1e-6)  # 验证根和数值误差范围。
