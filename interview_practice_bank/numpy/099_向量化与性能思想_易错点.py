"""
题目 099：向量化与性能思想_易错点

要求：完成“向量化与性能思想”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建待计算数据。
2. 使用ufunc向量化完成逐元素公式。
3. vectorize只是便利循环，并不会自动获得真正ufunc性能。

完成标准：
- 验证语义相同但性能含义不同。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.arange(1, 1001, dtype=float)  # 创建待计算数据。
vectorized = values**2 + 2 * values + 1  # 使用ufunc向量化完成逐元素公式。
python_wrapper = np.vectorize(lambda value: value**2 + 2 * value + 1); wrapped = python_wrapper(values)  # vectorize只是便利循环，并不会自动获得真正ufunc性能。
assert np.array_equal(wrapped, vectorized)  # 验证语义相同但性能含义不同。
