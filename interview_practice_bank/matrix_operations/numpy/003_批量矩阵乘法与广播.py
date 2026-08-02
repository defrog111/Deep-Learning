"""
题目 003：NumPy 批量矩阵乘法与广播

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 创建一批矩阵。
2. 让共享权重矩阵沿 batch 维广播。
3. 再完成两个 batch 的逐批矩阵乘法。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
batch = np.arange(24.0).reshape(2, 3, 4)  # 创建 shape=(B=2,M=3,K=4) 的批量矩阵。
shared_weight = np.ones((4, 5))  # 创建所有 batch 共用的 shape=(K=4,N=5) 权重。
broadcast_result = batch @ shared_weight  # matmul 自动广播共享权重得到 (2,3,5)。
batch_weight = np.ones((2, 4, 5))  # 为每个 batch 创建独立权重。
batch_result = np.matmul(batch, batch_weight)  # 逐 batch 执行 (3,4)@(4,5)。
assert broadcast_result.shape == batch_result.shape == (2, 3, 5)  # 验证批量输出 shape。
assert np.allclose(broadcast_result, batch_result)  # 当前权重相同，因此两个结果应相等。
print(broadcast_result.shape, broadcast_result[0])  # 输出 shape 和第一个 batch。
