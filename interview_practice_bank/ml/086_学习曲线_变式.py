"""
题目 086：学习曲线_变式

要求：完成“学习曲线”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 随数据增加间距缩小说明继续加数据可能有帮助。
3. 验证高方差学习曲线。

完成标准：
- 验证高方差学习曲线。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
sample_sizes = np.array([20, 50, 100, 200]); train_curve = np.array([0.99, 0.95, 0.9, 0.86]); valid_curve = np.array([0.55, 0.65, 0.75, 0.82]); shrinking_gap = train_curve - valid_curve  # 随数据增加间距缩小说明继续加数据可能有帮助。
assert shrinking_gap[-1] < shrinking_gap[0]  # 验证高方差学习曲线。
