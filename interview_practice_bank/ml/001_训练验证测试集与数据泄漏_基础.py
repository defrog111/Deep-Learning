"""
题目 001：训练验证测试集与数据泄漏_基础

要求：完成“训练验证测试集与数据泄漏”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 固定切分随机性。
2. 一步生成0到99的不重复随机排列。
3. 按60/20/20切分。
4. 构造单特征数据。
5. 只在训练集拟合预处理统计量。
6. 用训练统计量转换测试集。

完成标准：
- 验证集合互斥和测试样本数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 固定切分随机性。
indices = rng.permutation(100)  # 一步生成0到99的不重复随机排列。
train, valid, test = indices[:60], indices[60:80], indices[80:]  # 按60/20/20切分。
features = np.arange(100, dtype=float)  # 构造单特征数据。
train_mean = features[train].mean()  # 只在训练集拟合预处理统计量。
standardized_test = features[test] - train_mean  # 用训练统计量转换测试集。
assert not set(train) & set(test) and len(standardized_test) == 20  # 验证集合互斥和测试样本数。
print(train_mean, standardized_test[:3])  # 输出无泄漏处理结果。
# -------------------- 其他写法（以下代码作为知识补充，不会执行） --------------------
# 写法A：先创建连续索引，再原地shuffle；结果和permutation一样是不重复的全排列。
# alternative_indices = np.arange(100)  # 创建0到99且不重复的索引。
# rng.shuffle(alternative_indices)  # 原地打乱；注意该方法会修改原数组。
# 写法B：使用choice且设置replace=False，也能进行无放回抽样。
# alternative_indices = rng.choice(100, size=100, replace=False)  # 无放回抽取全部索引。
# 写法C：如果只需要随机抽取20个测试索引，可以不生成完整排列。
# test_indices = rng.choice(100, size=20, replace=False)  # 直接无放回抽取测试集索引。
# 易错写法：rng.integers(0, 100, size=100)默认有放回，因此会重复并遗漏部分索引。
# wrong_indices = rng.integers(0, 100, size=100)  # 这是随机整数采样，不是随机排列。
# assert np.unique(wrong_indices).size <= 100  # unique数量通常小于100，不能直接拿来切分数据集。
