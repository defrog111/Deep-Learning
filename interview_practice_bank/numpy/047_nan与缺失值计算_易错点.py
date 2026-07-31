"""
题目 047：NaN与缺失值计算_易错点

要求：完成“NaN与缺失值计算”的易错点题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])  # 构造含NaN数组。
mask = np.isnan(values)  # 识别NaN而不能用等号比较。
mean = np.nanmean(values)  # 忽略NaN计算均值。
filled = np.nan_to_num(values, nan=mean)  # 使用均值替换NaN。
assert not np.isnan(filled).any()  # 验证填充完成。
print(mask, mean, filled)  # 输出缺失值处理结果。
