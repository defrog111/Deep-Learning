"""
题目 023：广播机制_易错点

要求：完成“广播机制”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 主动运行一个不兼容的广播操作。
    np.ones((2, 3)) + np.ones((2, 2))  # 尝试广播不兼容的尾部维度。
except ValueError as error:  # 捕获预期的广播异常。
    broadcast_error = str(error)  # 保存shape不兼容错误。
3. 验证错误来自广播规则。

完成标准：
- 验证错误来自广播规则。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
try:  # 主动运行一个不兼容的广播操作。
    np.ones((2, 3)) + np.ones((2, 2))  # 尝试广播不兼容的尾部维度。
except ValueError as error:  # 捕获预期的广播异常。
    broadcast_error = str(error)  # 保存shape不兼容错误。
assert 'broadcast' in broadcast_error.lower()  # 验证错误来自广播规则。
