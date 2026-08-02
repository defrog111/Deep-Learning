"""
题目 052：随机数Generator与复现_综合

要求：完成“随机数Generator与复现”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 用SeedSequence为并行任务创建独立随机流。
3. 验证子随机流不同。

完成标准：
- 验证子随机流不同。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
seed_sequence = np.random.SeedSequence(42); child_a, child_b = [np.random.default_rng(seed) for seed in seed_sequence.spawn(2)]  # 用SeedSequence为并行任务创建独立随机流。
assert not np.array_equal(child_a.normal(size=5), child_b.normal(size=5))  # 验证子随机流不同。
