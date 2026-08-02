"""
题目 011：线性回归正规方程_易错点

要求：完成“线性回归正规方程”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 完全共线使XTX不可逆，应使用lstsq或伪逆。
3. 验证多重共线性陷阱。

完成标准：
- 验证多重共线性陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
collinear = np.c_[np.arange(5.0), 2 * np.arange(5.0)]; collinear_rank = np.linalg.matrix_rank(collinear); stable_weights = np.linalg.pinv(collinear) @ np.arange(5.0)  # 完全共线使XTX不可逆，应使用lstsq或伪逆。
assert collinear_rank == 1 and np.isfinite(stable_weights).all()  # 验证多重共线性陷阱。
