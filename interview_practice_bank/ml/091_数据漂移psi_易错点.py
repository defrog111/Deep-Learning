"""
题目 091：数据漂移PSI_易错点

要求：完成“数据漂移PSI”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. PSI遇到零箱会产生无穷，需平滑。
3. 验证零概率陷阱。

完成标准：
- 验证零概率陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
zero_reference = np.array([0.5, 0.5, 0.0]); nonzero_current = np.array([0.4, 0.4, 0.2]); unsafe_psi = np.isfinite((nonzero_current - zero_reference) * np.log(np.divide(nonzero_current, zero_reference, out=np.full(3, np.inf), where=zero_reference != 0))).all()  # PSI遇到零箱会产生无穷，需平滑。
assert not unsafe_psi  # 验证零概率陷阱。
