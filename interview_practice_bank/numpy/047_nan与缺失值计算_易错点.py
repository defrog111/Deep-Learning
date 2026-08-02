"""
题目 047：NaN与缺失值计算_易错点

要求：完成“NaN与缺失值计算”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 构造含NaN数组。
3. MaskedArray可同时记录缺失mask和数据。
4. 验证屏蔽无效值后的均值。

完成标准：
- 验证屏蔽无效值后的均值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])  # 构造含NaN数组。
masked = np.ma.masked_invalid(values); masked_mean = masked.mean()  # MaskedArray可同时记录缺失mask和数据。
assert np.isfinite(masked_mean)  # 验证屏蔽无效值后的均值。
