"""
题目 095：view与copy内存陷阱_易错点

要求：完成“view与copy内存陷阱”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建原始数组。
2. may_share_memory更快但可能保守判断。
3. 本例两个函数都识别出共享。

完成标准：
- 本例两个函数都识别出共享。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
original = np.arange(6)  # 创建原始数组。
possible_share = np.may_share_memory(original, original[::2]); definite_share = np.shares_memory(original, original[::2])  # may_share_memory更快但可能保守判断。
assert possible_share and definite_share  # 本例两个函数都识别出共享。
