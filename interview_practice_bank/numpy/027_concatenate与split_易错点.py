"""
题目 027：concatenate与split_易错点

要求：完成“concatenate与split”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. array_split允许不能整除，split则会抛错。
3. 验证不等长拆分规则。

完成标准：
- 验证不等长拆分规则。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
uneven = np.array_split(np.arange(10), 3)  # array_split允许不能整除，split则会抛错。
assert [len(part) for part in uneven] == [4, 3, 3]  # 验证不等长拆分规则。
