"""
题目 023：广播机制_易错点

要求：完成“广播机制”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建shape为(3,4)的矩阵。
2. 创建shape为(4,)的列方向偏置。
3. 显式变成shape为(3,1)的行方向缩放。
4. 利用广播完成无循环计算。
5. 广播结果保持矩阵shape。
6. 主动运行一个不兼容的广播操作。
    np.ones((2, 3)) + np.ones((2, 2))  # 尝试广播不兼容的尾部维度。
except ValueError as error:  # 捕获预期的广播异常。
    broadcast_error = str(error)  # 保存shape不兼容错误。

完成标准：
- 广播结果保持矩阵shape。
- 验证错误来自广播规则。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(12).reshape(3, 4)  # 创建shape为(3,4)的矩阵。
column_bias = np.array([10, 20, 30, 40])  # 创建shape为(4,)的列方向偏置。
row_scale = np.array([1, 2, 3]).reshape(3, 1)  # 显式变成shape为(3,1)的行方向缩放。
answer = (matrix + column_bias) * row_scale  # 利用广播完成无循环计算。
assert answer.shape == matrix.shape  # 广播结果保持矩阵shape。
print(answer)  # 输出广播运算结果。
try:  # 主动运行一个不兼容的广播操作。
    np.ones((2, 3)) + np.ones((2, 2))  # 尝试广播不兼容的尾部维度。
except ValueError as error:  # 捕获预期的广播异常。
    broadcast_error = str(error)  # 保存shape不兼容错误。
assert 'broadcast' in broadcast_error.lower()  # 验证错误来自广播规则。
