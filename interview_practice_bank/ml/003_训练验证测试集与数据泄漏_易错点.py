"""
题目 003：训练验证测试集与数据泄漏_易错点

要求：完成“训练验证测试集与数据泄漏”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造训练期特征。
2. 构造发生分布变化的测试期特征。
3. 正确做法只用训练集计算均值。
4. 错误做法偷看测试集后计算均值。
5. 使用训练统计量转换测试集。
6. 使用泄漏统计量会人为改变测试分布。
7. 衡量两种转换结果的平均差异。

完成标准：
- 验证测试信息确实污染了预处理参数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
train_features = np.arange(80, dtype=float)  # 构造训练期特征。
test_features = np.arange(1000, 1020, dtype=float)  # 构造发生分布变化的测试期特征。
train_mean = train_features.mean()  # 正确做法只用训练集计算均值。
leaked_mean = np.r_[train_features, test_features].mean()  # 错误做法偷看测试集后计算均值。
correct_test = test_features - train_mean  # 使用训练统计量转换测试集。
leaked_test = test_features - leaked_mean  # 使用泄漏统计量会人为改变测试分布。
difference = np.abs(correct_test - leaked_test).mean()  # 衡量两种转换结果的平均差异。
assert difference > 0 and train_mean != leaked_mean  # 验证测试信息确实污染了预处理参数。
print(train_mean, leaked_mean, difference)  # 输出正确与泄漏结果。
# -------------------- 其他写法（以下代码作为知识补充，不会执行） --------------------
# 写法A：StandardScaler同样必须先fit训练集，再分别transform训练集和测试集。
# from sklearn.preprocessing import StandardScaler  # 导入标准化器。
# scaler = StandardScaler().fit(train_features.reshape(-1, 1))  # 只在训练集拟合。
# scaled_test = scaler.transform(test_features.reshape(-1, 1))  # 使用训练统计量转换测试集。
# 错误写法：scaler.fit_transform(np.r_[train_features, test_features].reshape(-1, 1))会发生数据泄漏。
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
