"""
题目 096：view与copy内存陷阱_综合

要求：完成“view与copy内存陷阱”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建原始数组。
2. 基本切片通常返回共享内存的视图。
3. 花式索引返回独立副本。
4. 修改视图会影响原数组。
5. 修改副本不会影响原数组。
6. 综合把非连续转置结果变为C连续副本。

完成标准：
- 验证view与copy差异。
- 验证连续化产生独立存储。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
original = np.arange(6)  # 创建原始数组。
view = original[1:4]  # 基本切片通常返回共享内存的视图。
copy = original[[1, 2, 3]]  # 花式索引返回独立副本。
view[0] = 99  # 修改视图会影响原数组。
copy[1] = 88  # 修改副本不会影响原数组。
assert original[1] == 99 and original[2] != 88  # 验证view与copy差异。
print(original, view, copy, np.shares_memory(original, view))  # 输出内存共享信息。
transposed = original.reshape(2, 3).T; contiguous = np.ascontiguousarray(transposed)  # 综合把非连续转置结果变为C连续副本。
assert contiguous.flags.c_contiguous and not np.shares_memory(contiguous, original)  # 验证连续化产生独立存储。
