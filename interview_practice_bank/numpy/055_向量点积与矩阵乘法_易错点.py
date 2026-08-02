"""
题目 055：向量点积与矩阵乘法_易错点

要求：完成“向量点积与矩阵乘法”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. matmul支持批量矩阵乘法。
3. 验证批量维广播。

完成标准：
- 验证批量维广播。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
batched_left = np.ones((2, 3, 4)); batched_right = np.ones((2, 4, 5)); batched_product = np.matmul(batched_left, batched_right)  # matmul支持批量矩阵乘法。
assert batched_product.shape == (2, 3, 5)  # 验证批量维广播。
