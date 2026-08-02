"""
题目 079：outer与einsum_易错点

要求：完成“outer与einsum”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. kron计算Kronecker积，不能和普通外积混淆。
3. 验证Kronecker积shape。

完成标准：
- 验证Kronecker积shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
kronecker = np.kron(np.eye(2), np.ones((2, 2)))  # kron计算Kronecker积，不能和普通外积混淆。
assert kronecker.shape == (4, 4)  # 验证Kronecker积shape。
