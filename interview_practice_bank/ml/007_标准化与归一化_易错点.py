"""
题目 007：标准化与归一化_易错点

要求：完成“标准化与归一化”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建尺度差异明显的特征。
2. 计算总体均值和标准差。
3. Z-score标准化使均值0方差1。
4. 计算每列极值。
5. Min-Max归一化到[0,1]。
6. 常数列标准差为零，必须防止除零。

完成标准：
- 验证缩放性质。
- 验证安全处理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 40.0]])  # 创建尺度差异明显的特征。
mean, std = matrix.mean(0), matrix.std(0)  # 计算总体均值和标准差。
standardized = (matrix - mean) / std  # Z-score标准化使均值0方差1。
minimum, maximum = matrix.min(0), matrix.max(0)  # 计算每列极值。
normalized = (matrix - minimum) / (maximum - minimum)  # Min-Max归一化到[0,1]。
assert np.allclose(standardized.mean(0), 0) and np.allclose(normalized.min(0), 0)  # 验证缩放性质。
print(standardized, '\n', normalized)  # 输出两种缩放。
constant_feature = np.ones(4); safe_denominator = np.where(constant_feature.std() == 0, 1.0, constant_feature.std()); safe_standardized = (constant_feature - constant_feature.mean()) / safe_denominator  # 常数列标准差为零，必须防止除零。
assert np.isfinite(safe_standardized).all() and np.allclose(safe_standardized, 0)  # 验证安全处理。
