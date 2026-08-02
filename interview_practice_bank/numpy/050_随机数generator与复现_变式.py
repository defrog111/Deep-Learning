"""
题目 050：随机数Generator与复现_变式

要求：完成“随机数Generator与复现”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 使用推荐的Generator并固定种子。
3. permutation、arange加shuffle、choice无放回均得到不重复索引。
4. 验证三种随机排列写法。

完成标准：
- 验证三种随机排列写法。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 使用推荐的Generator并固定种子。
permuted = rng.permutation(100); shuffled = np.arange(100); rng.shuffle(shuffled); chosen = rng.choice(100, size=100, replace=False)  # permutation、arange加shuffle、choice无放回均得到不重复索引。
assert len(np.unique(permuted)) == len(np.unique(shuffled)) == len(np.unique(chosen)) == 100  # 验证三种随机排列写法。
