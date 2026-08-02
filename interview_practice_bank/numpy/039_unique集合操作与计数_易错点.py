"""
题目 039：unique集合操作与计数_易错点

要求：完成“unique集合操作与计数”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 使用集合运算并注意结果会排序。
3. 验证集合结果。

完成标准：
- 验证集合结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
union = np.union1d([1, 2, 3], [3, 4]); difference = np.setdiff1d([1, 2, 3], [2])  # 使用集合运算并注意结果会排序。
assert union.tolist() == [1, 2, 3, 4] and difference.tolist() == [1, 3]  # 验证集合结果。
