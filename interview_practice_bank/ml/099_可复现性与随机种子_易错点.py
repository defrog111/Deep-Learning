"""
题目 099：可复现性与随机种子_易错点

要求：完成“可复现性与随机种子”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入Python随机模块。
2. 导入 NumPy。
3. 相同seed复现实验，不同seed用于评估方差。
4. 验证seed语义。

完成标准：
- 验证seed语义。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import random  # 导入Python随机模块。
import numpy as np  # 导入 NumPy。
same_seed_predictions = [np.random.default_rng(42).integers(0, 2, size=10) for _ in range(2)]; different_seed_predictions = [np.random.default_rng(seed).integers(0, 2, size=10) for seed in [42, 43]]  # 相同seed复现实验，不同seed用于评估方差。
assert np.array_equal(*same_seed_predictions) and not np.array_equal(*different_seed_predictions)  # 验证seed语义。
