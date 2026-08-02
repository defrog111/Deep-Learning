"""
题目 046：NaN与缺失值计算_变式

要求：完成“NaN与缺失值计算”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 构造含NaN数组。
3. 忽略NaN计算中位数和分位数。
4. 验证NaN安全统计。

完成标准：
- 验证NaN安全统计。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])  # 构造含NaN数组。
median = np.nanmedian(values); percentile = np.nanpercentile(values, [25, 75])  # 忽略NaN计算中位数和分位数。
assert np.isfinite(median) and percentile.shape == (2,)  # 验证NaN安全统计。
