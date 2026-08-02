"""
题目 031：聚合函数与axis_易错点

要求：完成“聚合函数与axis”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 指定过小累加dtype会发生整数溢出。
3. 对比错误与安全累加dtype。

完成标准：
- 对比错误与安全累加dtype。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
small = np.array([250, 10], dtype=np.uint8); overflowed = small.sum(dtype=np.uint8)  # 指定过小累加dtype会发生整数溢出。
assert overflowed == 4 and small.sum(dtype=np.int64) == 260  # 对比错误与安全累加dtype。
