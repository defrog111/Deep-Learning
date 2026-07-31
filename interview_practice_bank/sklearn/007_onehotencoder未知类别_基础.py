"""
题目 007：OneHotEncoder未知类别_基础

要求：完成“OneHotEncoder未知类别”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import OneHotEncoder  # 导入独热编码器。
train = np.array([['red'], ['blue'], ['red']])  # 创建训练类别。
test = np.array([['green'], ['red']])  # 测试集包含未知类别。
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(train)  # 忽略推理时未知类别避免报错。
encoded = encoder.transform(test)  # 转换测试类别。
assert encoded.shape == (2, 2) and encoded[0].sum() == 0  # 未知类别编码为全零。
print(encoder.categories_, encoded)  # 输出学习到的类别和编码。
