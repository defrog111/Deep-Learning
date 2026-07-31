"""
题目 100：可复现性与随机种子_综合

要求：完成“可复现性与随机种子”的综合题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import random  # 导入Python随机模块。
import numpy as np  # 导入 NumPy。
seed = 44  # 设置统一随机种子。
random.seed(seed)  # 固定Python随机性。
np_rng = np.random.default_rng(seed)  # 创建局部NumPy Generator。
python_values = [random.random() for _ in range(3)]  # 采样Python随机数。
numpy_values = np_rng.normal(size=3)  # 采样NumPy随机数。
random.seed(seed)  # 重置Python种子验证复现。
assert python_values == [random.random() for _ in range(3)]  # 验证Python随机序列复现。
assert np.allclose(numpy_values, np.random.default_rng(seed).normal(size=3))  # 验证NumPy随机序列复现。
print(seed, python_values, numpy_values)  # 输出可复现样本。
