"""
题目 076：范数与向量距离_综合

要求：完成“范数与向量距离”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建向量a。
2. 创建向量b。
3. 计算曼哈顿距离。
4. 计算欧氏距离。
5. 计算余弦相似度。

完成标准：
- 验证余弦范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1.0, 2.0, 3.0])  # 创建向量a。
b = np.array([4.0, 2.0, 0.0])  # 创建向量b。
l1 = np.linalg.norm(a - b, ord=1)  # 计算曼哈顿距离。
l2 = np.linalg.norm(a - b, ord=2)  # 计算欧氏距离。
cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))  # 计算余弦相似度。
assert -1 <= cosine <= 1  # 验证余弦范围。
print(l1, l2, cosine)  # 输出三种相似性指标。
