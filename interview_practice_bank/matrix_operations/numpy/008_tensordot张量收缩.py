"""
题目 008：NumPy tensordot张量收缩

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 指定两个张量要收缩的轴。
2. 预测未收缩轴组成的输出 shape。
3. 用 einsum 写出等价形式。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
left = np.arange(24.0).reshape(2, 3, 4)  # 创建 shape=(2,3,4) 的左张量。
right = np.arange(20.0).reshape(4, 5)  # 创建 shape=(4,5) 的右张量。
contracted = np.tensordot(left, right, axes=([2], [0]))  # 收缩 left 轴2与 right 轴0。
via_einsum = np.einsum('ijk,kl->ijl', left, right)  # 显式标注相同的 k 轴收缩。
assert contracted.shape == (2, 3, 5) and np.allclose(contracted, via_einsum)  # 验证输出。
print(contracted.shape, contracted[0])  # 输出 shape 和第一个切片。
