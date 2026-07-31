"""
题目 008：OneHotEncoder未知类别_变式

要求：完成“OneHotEncoder未知类别”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建训练类别。
2. 测试集包含未知类别。
3. 忽略推理时未知类别避免报错。
4. 转换测试类别。
5. 未知类别编码为全零。

完成标准：
- 未知类别编码为全零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import OneHotEncoder  # 导入独热编码器。
train = np.array([['red'], ['blue'], ['red']])  # 创建训练类别。
test = np.array([['green'], ['red']])  # 测试集包含未知类别。
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(train)  # 忽略推理时未知类别避免报错。
encoded = encoder.transform(test)  # 转换测试类别。
assert encoded.shape == (2, 2) and encoded[0].sum() == 0  # 未知类别编码为全零。
print(encoder.categories_, encoded)  # 输出学习到的类别和编码。
