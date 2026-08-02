"""
题目 005：NumPy inner_outer与kronecker积

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算内积和外积。
2. 计算两个小矩阵的 Kronecker 积。
3. 根据 shape 解释三者差异。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1.0, 2.0, 3.0])  # 创建第一个向量。
b = np.array([4.0, 5.0])  # 创建不同长度的第二个向量。
inner = np.inner(a, a)  # inner 对最后轴收缩并得到标量。
outer = np.outer(a, b)  # outer 组合每一对元素并得到 shape=(3,2)。
kronecker = np.kron(np.eye(2), np.ones((2, 2)))  # 计算分块式 Kronecker 积。
assert inner == 14.0 and outer.shape == (3, 2) and kronecker.shape == (4, 4)  # 核对 shape。
print(inner, outer, kronecker)  # 输出三种结果。
