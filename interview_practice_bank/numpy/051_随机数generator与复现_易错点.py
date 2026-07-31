"""
题目 051：随机数Generator与复现_易错点

要求：完成“随机数Generator与复现”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 使用推荐的Generator并固定种子。
2. 从正态分布采样。
3. 从离散均匀分布采样。
4. 按概率采样类别。
5. 用相同种子复现第一组采样。

完成标准：
- 验证可复现性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(43)  # 使用推荐的Generator并固定种子。
normal = rng.normal(loc=0.0, scale=1.0, size=(2, 3))  # 从正态分布采样。
integers = rng.integers(0, 10, size=5)  # 从离散均匀分布采样。
choice = rng.choice(['A', 'B', 'C'], size=4, replace=True, p=[0.2, 0.3, 0.5])  # 按概率采样类别。
repeat = np.random.default_rng(43).normal(size=(2, 3))  # 用相同种子复现第一组采样。
assert np.allclose(normal, repeat)  # 验证可复现性。
print(normal, integers, choice)  # 输出随机样本。
