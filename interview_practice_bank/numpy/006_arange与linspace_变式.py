"""
题目 006：arange与linspace_变式

要求：完成“arange与linspace”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. arange三种等价常用写法。
3. 验证三种参数形式结果一致。

完成标准：
- 验证三种参数形式结果一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
range_short = np.arange(100); range_full = np.arange(0, 100); range_keywords = np.arange(start=0, stop=100, step=1)  # arange三种等价常用写法。
assert np.array_equal(range_short, range_full) and np.array_equal(range_full, range_keywords)  # 验证三种参数形式结果一致。
