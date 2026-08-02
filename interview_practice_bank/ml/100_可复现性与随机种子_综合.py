"""
题目 100：可复现性与随机种子_综合

要求：完成“可复现性与随机种子”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入Python随机模块。
2. 导入 NumPy。
3. 综合复现还需代码、数据、特征和指标版本，不能只有随机种子。
4. 验证实验追踪元数据。

完成标准：
- 验证实验追踪元数据。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import random  # 导入Python随机模块。
import numpy as np  # 导入 NumPy。
experiment_record = {'code_version': 'abc123', 'data_version': 'v2', 'seed': 42, 'features': ['x1', 'x2'], 'metric': 0.91}; required_fields = {'code_version', 'data_version', 'seed', 'features', 'metric'}  # 综合复现还需代码、数据、特征和指标版本，不能只有随机种子。
assert required_fields <= experiment_record.keys()  # 验证实验追踪元数据。
