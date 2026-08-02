"""
题目 007：arange与linspace_易错点

要求：完成“arange与linspace”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 浮点步长用arange可能累积误差，固定点数优先linspace。
3. 比较左闭右开与包含终点。

完成标准：
- 比较左闭右开与包含终点。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
float_steps = np.arange(0.0, 1.0, 0.1); fixed_points = np.linspace(0.0, 1.0, 11)  # 浮点步长用arange可能累积误差，固定点数优先linspace。
assert len(float_steps) == 10 and len(fixed_points) == 11 and fixed_points[-1] == 1  # 比较左闭右开与包含终点。
