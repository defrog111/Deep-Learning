"""
题目 100：向量化与性能思想_综合

要求：完成“向量化与性能思想”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建待计算数据。
2. 综合使用ufunc的out和where条件写入。
3. 本例所有值为正，因此等于完整平方。

完成标准：
- 本例所有值为正，因此等于完整平方。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.arange(1, 1001, dtype=float)  # 创建待计算数据。
positive_only = np.zeros_like(values); np.square(values, out=positive_only, where=values > 0)  # 综合使用ufunc的out和where条件写入。
assert np.array_equal(positive_only, values**2)  # 本例所有值为正，因此等于完整平方。
