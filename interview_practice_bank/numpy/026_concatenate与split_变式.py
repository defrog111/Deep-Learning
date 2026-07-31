"""
题目 026：concatenate与split_变式

要求：完成“concatenate与split”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建左侧矩阵。
2. 创建右侧矩阵。
3. 沿行轴拼接。
4. 沿列轴拼接。
5. 允许不等分地拆成三块。
6. 比较新增轴stack与已有轴拼接。

完成标准：
- 验证不同轴的shape。
- 验证三种拼接。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
left = np.arange(6).reshape(2, 3)  # 创建左侧矩阵。
right = np.arange(6, 12).reshape(2, 3)  # 创建右侧矩阵。
rows = np.concatenate([left, right], axis=0)  # 沿行轴拼接。
columns = np.concatenate([left, right], axis=1)  # 沿列轴拼接。
parts = np.array_split(columns, 3, axis=1)  # 允许不等分地拆成三块。
assert rows.shape == (4, 3) and columns.shape == (2, 6)  # 验证不同轴的shape。
print(rows, '\n', [part.shape for part in parts])  # 输出拼接与拆分结果。
stacked = np.stack([left, right], axis=0); vertical = np.vstack([left, right]); horizontal = np.hstack([left, right])  # 比较新增轴stack与已有轴拼接。
assert stacked.shape == (2, 2, 3) and vertical.shape == rows.shape and horizontal.shape == columns.shape  # 验证三种拼接。
