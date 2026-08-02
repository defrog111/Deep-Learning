"""
题目 095：时间序列切分_易错点

要求：完成“时间序列切分”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 普通随机切分破坏时间因果顺序。
3. 验证时间序列不能随意shuffle。

完成标准：
- 验证时间序列不能随意shuffle。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
random_split = np.random.default_rng(0).permutation(np.arange(20)); random_train, random_test = random_split[:15], random_split[15:]; chronological_violation = random_train.max() > random_test.min()  # 普通随机切分破坏时间因果顺序。
assert chronological_violation  # 验证时间序列不能随意shuffle。
