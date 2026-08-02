"""
题目 002：NumPy multiply_dot与matmul区别

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 比较逐元素乘法与矩阵乘法。
2. 比较二维数组上的 dot、matmul 和 @。
3. 说明不能把星号当矩阵乘法。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
left = np.array([[1.0, 2.0], [3.0, 4.0]])  # 创建左矩阵。
right = np.array([[2.0, 0.0], [1.0, 2.0]])  # 创建右矩阵。
elementwise = np.multiply(left, right)  # multiply或星号执行逐位置相乘。
via_at = left @ right  # @ 执行矩阵乘法。
via_dot = np.dot(left, right)  # 二维输入时 dot 与矩阵乘法一致。
via_matmul = np.matmul(left, right)  # matmul 与 @ 的语义一致。
assert not np.array_equal(elementwise, via_at)  # 验证逐元素乘法不是矩阵乘法。
assert np.allclose(via_at, via_dot) and np.allclose(via_at, via_matmul)  # 核对三种矩阵乘法写法。
print(elementwise, via_at)  # 对比两类结果。
