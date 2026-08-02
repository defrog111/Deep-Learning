"""
题目 021：NumPy 矩阵标准化与广播

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 按列计算均值和标准差。
2. 使用 keepdims 保留广播所需维度。
3. 标准化后检查每列均值和总体标准差。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])  # 行是样本、列是特征。
mean = matrix.mean(axis=0, keepdims=True)  # 沿样本轴求每列均值并保留 shape=(1,2)。
std = matrix.std(axis=0, keepdims=True, ddof=0)  # 计算总体标准差并保留广播维度。
standardized = (matrix - mean) / std  # 借助广播一次标准化所有列。
assert mean.shape == std.shape == (1, 2)  # 验证 keepdims 的作用。
assert np.allclose(standardized.mean(axis=0), 0.0)  # 验证每列均值为零。
assert np.allclose(standardized.std(axis=0), 1.0)  # 验证每列总体标准差为一。
print(mean, std, standardized)  # 输出标准化参数和结果。
