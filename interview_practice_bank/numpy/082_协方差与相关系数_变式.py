"""
题目 082：协方差与相关系数_变式

要求：完成“协方差与相关系数”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 每行一个样本每列一个特征。
3. 样本在行、特征在列时必须设置rowvar=False。
4. 验证特征协方差矩阵shape。

完成标准：
- 验证特征协方差矩阵shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])  # 每行一个样本每列一个特征。
samples_by_row = np.cov(samples, rowvar=False, ddof=1)  # 样本在行、特征在列时必须设置rowvar=False。
assert samples_by_row.shape == (samples.shape[1], samples.shape[1])  # 验证特征协方差矩阵shape。
