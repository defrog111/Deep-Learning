"""
题目 049：随机数Generator与复现_基础

要求：完成“随机数Generator与复现”的基础题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(41)  # 使用推荐的Generator并固定种子。
normal = rng.normal(loc=0.0, scale=1.0, size=(2, 3))  # 从正态分布采样。
integers = rng.integers(0, 10, size=5)  # 从离散均匀分布采样。
choice = rng.choice(['A', 'B', 'C'], size=4, replace=True, p=[0.2, 0.3, 0.5])  # 按概率采样类别。
repeat = np.random.default_rng(41).normal(size=(2, 3))  # 用相同种子复现第一组采样。
assert np.allclose(normal, repeat)  # 验证可复现性。
print(normal, integers, choice)  # 输出随机样本。
