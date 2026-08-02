"""
题目 014：NumPy svd与低秩近似

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 对非方阵做经济型 SVD。
2. 重建原矩阵。
3. 只保留最大奇异值构造 rank-1 近似并比较误差。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])  # 创建 2×3 非方阵。
u, singular, vh = np.linalg.svd(matrix, full_matrices=False)  # 计算经济型 A=UΣVh。
reconstructed = (u * singular) @ vh  # 利用广播把 U 每列乘对应奇异值。
rank_one = (u[:, :1] * singular[:1]) @ vh[:1]  # 只保留最大奇异值构造最佳 rank-1 近似。
full_error = np.linalg.norm(matrix - reconstructed)  # 计算完整重建误差。
rank_one_error = np.linalg.norm(matrix - rank_one)  # 计算压缩后的 Frobenius 误差。
assert full_error < 1e-10 and rank_one_error > full_error  # 验证完整重建和有损近似。
print(singular, rank_one, rank_one_error)  # 输出奇异值与低秩近似。
