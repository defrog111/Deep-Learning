"""
题目 084：协方差与相关系数_综合

要求：完成“协方差与相关系数”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 每行一个样本每列一个特征。
3. 标准化后的协方差等于相关矩阵。
4. 验证协方差与相关系数关系。

完成标准：
- 验证协方差与相关系数关系。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])  # 每行一个样本每列一个特征。
standardized_matrix = (samples - samples.mean(0)) / samples.std(0, ddof=1); correlation_from_cov = np.cov(standardized_matrix, rowvar=False)  # 标准化后的协方差等于相关矩阵。
assert np.allclose(correlation_from_cov, np.corrcoef(samples, rowvar=False))  # 验证协方差与相关系数关系。
