"""
题目 052：随机数Generator与复现_综合

要求：完成“随机数Generator与复现”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 使用推荐的Generator并固定种子。
2. 从正态分布采样。
3. 从离散均匀分布采样。
4. 按概率采样类别。
5. 用相同种子复现第一组采样。
6. 用SeedSequence为并行任务创建独立随机流。

完成标准：
- 验证可复现性。
- 验证子随机流不同。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(44)  # 使用推荐的Generator并固定种子。
normal = rng.normal(loc=0.0, scale=1.0, size=(2, 3))  # 从正态分布采样。
integers = rng.integers(0, 10, size=5)  # 从离散均匀分布采样。
choice = rng.choice(['A', 'B', 'C'], size=4, replace=True, p=[0.2, 0.3, 0.5])  # 按概率采样类别。
repeat = np.random.default_rng(44).normal(size=(2, 3))  # 用相同种子复现第一组采样。
assert np.allclose(normal, repeat)  # 验证可复现性。
print(normal, integers, choice)  # 输出随机样本。
seed_sequence = np.random.SeedSequence(42); child_a, child_b = [np.random.default_rng(seed) for seed in seed_sequence.spawn(2)]  # 用SeedSequence为并行任务创建独立随机流。
assert not np.array_equal(child_a.normal(size=5), child_b.normal(size=5))  # 验证子随机流不同。
