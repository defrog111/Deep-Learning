"""
题目 096：view与copy内存陷阱_综合

要求：完成“view与copy内存陷阱”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建原始数组。
3. 综合把非连续转置结果变为C连续副本。
4. 验证连续化产生独立存储。

完成标准：
- 验证连续化产生独立存储。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
original = np.arange(6)  # 创建原始数组。
transposed = original.reshape(2, 3).T; contiguous = np.ascontiguousarray(transposed)  # 综合把非连续转置结果变为C连续副本。
assert contiguous.flags.c_contiguous and not np.shares_memory(contiguous, original)  # 验证连续化产生独立存储。
