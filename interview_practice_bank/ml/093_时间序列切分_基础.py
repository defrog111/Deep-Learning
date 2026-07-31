"""
题目 093：时间序列切分_基础

要求：完成“时间序列切分”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建按时间排序样本。
2. 设置滚动验证窗口大小。
3. 保存时间序列切分。
4. 逐步扩展训练窗口。
5. 训练只能使用过去。
6. 保存本折。

完成标准：
- 验证没有未来信息泄漏。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
time = np.arange(20)  # 创建按时间排序样本。
test_size = 3  # 设置滚动验证窗口大小。
splits = []  # 保存时间序列切分。
for train_end in range(8, 20 - test_size + 1, test_size):  # 逐步扩展训练窗口。
    train = time[:train_end]  # 训练只能使用过去。
    validation = time[train_end:train_end + test_size]  # 验证使用紧随其后的未来。
    splits.append((train, validation))  # 保存本折。
assert all(train.max() < validation.min() for train, validation in splits)  # 验证没有未来信息泄漏。
print([(len(train), validation.tolist()) for train, validation in splits])  # 输出时间切分。
