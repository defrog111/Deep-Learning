"""
题目 100：向量化与性能思想_综合

要求：完成“向量化与性能思想”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建待计算数据。
2. 使用ufunc向量化完成逐元素公式。
3. 用Python循环表达相同逻辑用于对照。
4. 创建随变式变化的方阵。
5. 无显式循环计算每行L2范数。
6. 综合使用ufunc的out和where条件写入。
7. 本例所有值为正，因此等于完整平方。

完成标准：
- 验证向量化不改变结果。
- 本例所有值为正，因此等于完整平方。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.arange(1, 1001, dtype=float)  # 创建待计算数据。
vectorized = values**2 + 2 * values + 1  # 使用ufunc向量化完成逐元素公式。
loop_result = np.array([value**2 + 2 * value + 1 for value in values])  # 用Python循环表达相同逻辑用于对照。
matrix = np.arange(36).reshape(6, 6)  # 创建随变式变化的方阵。
row_norms = np.sqrt((matrix.astype(float) ** 2).sum(axis=1))  # 无显式循环计算每行L2范数。
assert np.array_equal(vectorized, loop_result)  # 验证向量化不改变结果。
print(vectorized[:3], row_norms)  # 输出部分结果。
positive_only = np.zeros_like(values); np.square(values, out=positive_only, where=values > 0)  # 综合使用ufunc的out和where条件写入。
assert np.array_equal(positive_only, values**2)  # 本例所有值为正，因此等于完整平方。
