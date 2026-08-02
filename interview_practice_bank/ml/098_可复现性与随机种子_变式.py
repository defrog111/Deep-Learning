"""
题目 098：可复现性与随机种子_变式

要求：完成“可复现性与随机种子”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入Python随机模块。
2. 导入 NumPy。
3. 并行任务应派生独立随机流而不是共用全局状态。
4. 验证可复现但相互独立。

完成标准：
- 验证可复现但相互独立。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import random  # 导入Python随机模块。
import numpy as np  # 导入 NumPy。
base_seed = 42; child_sequences = np.random.SeedSequence(base_seed).spawn(2); child_values = [np.random.default_rng(sequence).normal(size=3) for sequence in child_sequences]  # 并行任务应派生独立随机流而不是共用全局状态。
assert not np.array_equal(child_values[0], child_values[1])  # 验证可复现但相互独立。
