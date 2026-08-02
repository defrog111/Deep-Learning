"""
题目 048：NaN与缺失值计算_综合

要求：完成“NaN与缺失值计算”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 构造含NaN数组。
3. 显式指定NaN和正负无穷替换值。
4. 验证清洗后均为有限值。

完成标准：
- 验证清洗后均为有限值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])  # 构造含NaN数组。
cleaned = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)  # 显式指定NaN和正负无穷替换值。
assert np.isfinite(cleaned).all()  # 验证清洗后均为有限值。
