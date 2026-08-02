"""
题目 032：聚合函数与axis_综合

要求：完成“聚合函数与axis”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 综合使用ufunc的accumulate和reduce。
3. 验证累计和连乘。

完成标准：
- 验证累计和连乘。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
accumulated = np.add.accumulate(np.arange(1, 6)); reduced = np.multiply.reduce(np.arange(1, 6))  # 综合使用ufunc的accumulate和reduce。
assert accumulated[-1] == 15 and reduced == 120  # 验证累计和连乘。
