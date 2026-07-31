"""
题目 046：NaN与缺失值计算_变式

要求：完成“NaN与缺失值计算”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 构造含NaN数组。
2. 识别NaN而不能用等号比较。
3. 忽略NaN计算均值。
4. 使用均值替换NaN。

完成标准：
- 验证填充完成。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])  # 构造含NaN数组。
mask = np.isnan(values)  # 识别NaN而不能用等号比较。
mean = np.nanmean(values)  # 忽略NaN计算均值。
filled = np.nan_to_num(values, nan=mean)  # 使用均值替换NaN。
assert not np.isnan(filled).any()  # 验证填充完成。
print(mask, mean, filled)  # 输出缺失值处理结果。
