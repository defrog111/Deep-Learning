"""
题目 062：SVM间隔与hinge loss_变式

要求：完成“SVM间隔与hinge loss”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
weights = np.array([1.0, -1.0])  # 定义线性分类超平面法向量。
features = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.5]])  # 创建训练样本。
labels = np.array([1.0, -1.0, 1.0])  # SVM标签使用-1和1。
scores = features @ weights  # 计算到超平面的有符号分数。
margins = labels * scores  # 计算功能间隔。
hinge = np.maximum(0, 1 - margins)  # 计算hinge loss。
geometric_margin = margins / np.linalg.norm(weights)  # 除以权重范数得到几何间隔。
assert np.all(hinge >= 0)  # 验证损失非负。
print(scores, margins, hinge, geometric_margin)  # 输出SVM关键量。
