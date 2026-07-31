"""
题目 061：SVM间隔与hinge loss_基础

要求：完成“SVM间隔与hinge loss”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 定义线性分类超平面法向量。
2. 创建训练样本。
3. SVM标签使用-1和1。
4. 计算到超平面的有符号分数。
5. 计算功能间隔。
6. 计算hinge loss。
7. 除以权重范数得到几何间隔。

完成标准：
- 验证损失非负。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
