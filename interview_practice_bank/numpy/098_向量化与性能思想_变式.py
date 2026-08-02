"""
题目 098：向量化与性能思想_变式

要求：完成“向量化与性能思想”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建待计算数据。
2. 使用ufunc向量化完成逐元素公式。
3. 使用ufunc的out参数复用输出内存。

完成标准：
- 验证out写法结果一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.arange(1, 1001, dtype=float)  # 创建待计算数据。
vectorized = values**2 + 2 * values + 1  # 使用ufunc向量化完成逐元素公式。
output = np.empty_like(values); np.add(values**2, 2 * values + 1, out=output)  # 使用ufunc的out参数复用输出内存。
assert np.array_equal(output, vectorized)  # 验证out写法结果一致。
