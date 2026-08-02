"""
题目 051：随机数Generator与复现_易错点

要求：完成“随机数Generator与复现”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 使用推荐的Generator并固定种子。
2. integers默认允许重复，不能直接替代permutation。
3. 检查随机整数采样性质。

完成标准：
- 检查随机整数采样性质。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(43)  # 使用推荐的Generator并固定种子。
sampled_with_replacement = rng.integers(0, 100, size=100)  # integers默认允许重复，不能直接替代permutation。
assert sampled_with_replacement.shape == (100,) and np.unique(sampled_with_replacement).size <= 100  # 检查随机整数采样性质。
