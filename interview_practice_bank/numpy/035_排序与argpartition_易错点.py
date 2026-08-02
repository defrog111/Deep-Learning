"""
题目 035：排序与argpartition_易错点

要求：完成“排序与argpartition”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. stable排序保持相等元素原始先后。
3. 验证稳定性。

完成标准：
- 验证稳定性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
stable_order = np.argsort(np.array([2, 1, 2, 1]), kind='stable')  # stable排序保持相等元素原始先后。
assert stable_order.tolist() == [1, 3, 0, 2]  # 验证稳定性。
