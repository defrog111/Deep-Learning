"""
题目 007：NumPy einsum批量乘法与Attention分数

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 创建多头 Q 和 K。
2. 用 einsum 计算每个 head 的 QK 转置。
3. 与 matmul 写法核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 创建可复现随机数生成器。
query = rng.normal(size=(2, 3, 4, 5))  # 创建 (batch,head,query,key_dim) 的 Q。
key = rng.normal(size=(2, 3, 6, 5))  # 创建 (batch,head,key,key_dim) 的 K。
scores = np.einsum('bhqd,bhkd->bhqk', query, key)  # 收缩 d 并保留 batch、head、q、k。
via_matmul = query @ key.swapaxes(-1, -2)  # 使用批量 matmul 完成相同的 QK 转置。
assert scores.shape == (2, 3, 4, 6) and np.allclose(scores, via_matmul)  # 核对 shape 和数值。
print(scores.shape, scores[0, 0])  # 输出一个 head 的分数矩阵。
