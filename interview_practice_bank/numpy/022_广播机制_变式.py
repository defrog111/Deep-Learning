"""
题目 022：广播机制_变式

要求：完成“广播机制”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. broadcast_to创建只读广播视图而不复制数据。
3. 验证shape与只读属性。

完成标准：
- 验证shape与只读属性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
broadcasted = np.broadcast_to(np.arange(4), (3, 4))  # broadcast_to创建只读广播视图而不复制数据。
assert broadcasted.shape == (3, 4) and not broadcasted.flags.writeable  # 验证shape与只读属性。
