"""
题目 042：where与select条件选择_变式

要求：完成“where与select条件选择”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 构造跨多个条件区间的数据。
3. putmask原地修改满足条件的位置。
4. 验证掩码写入。

完成标准：
- 验证掩码写入。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([-3, 0, 2, 8, 12])  # 构造跨多个条件区间的数据。
masked_copy = values.copy(); np.putmask(masked_copy, masked_copy < 0, -1)  # putmask原地修改满足条件的位置。
assert np.all(masked_copy[values >= 0] >= 0)  # 验证掩码写入。
