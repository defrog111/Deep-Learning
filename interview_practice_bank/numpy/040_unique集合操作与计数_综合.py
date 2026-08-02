"""
题目 040：unique集合操作与计数_综合

要求：完成“unique集合操作与计数”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 非负小整数计数优先bincount。
3. 验证计数与补零。

完成标准：
- 验证计数与补零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
nonnegative = np.array([0, 1, 1, 3]); frequencies = np.bincount(nonnegative, minlength=5)  # 非负小整数计数优先bincount。
assert frequencies.tolist() == [1, 2, 0, 1, 0]  # 验证计数与补零。
