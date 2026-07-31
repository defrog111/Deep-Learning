"""
题目 002：训练验证测试集与数据泄漏_变式

要求：完成“训练验证测试集与数据泄漏”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 固定切分随机性。
2. 构造类别比例为80比20的不平衡标签。
3. 单独打乱类别0的样本索引。
4. 单独打乱类别1的样本索引。
5. 每类取60%组成训练集。
6. 每类取20%组成验证集。
7. 每类剩余20%组成测试集。
8. 计算三个集合的正类比例。

完成标准：
- 验证分层切分保持原始类别比例。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 固定切分随机性。
labels = np.array([0] * 80 + [1] * 20)  # 构造类别比例为80比20的不平衡标签。
class_zero = rng.permutation(np.flatnonzero(labels == 0))  # 单独打乱类别0的样本索引。
class_one = rng.permutation(np.flatnonzero(labels == 1))  # 单独打乱类别1的样本索引。
train = np.r_[class_zero[:48], class_one[:12]]  # 每类取60%组成训练集。
valid = np.r_[class_zero[48:64], class_one[12:16]]  # 每类取20%组成验证集。
test = np.r_[class_zero[64:], class_one[16:]]  # 每类剩余20%组成测试集。
ratios = [labels[part].mean() for part in (train, valid, test)]  # 计算三个集合的正类比例。
assert np.allclose(ratios, 0.2)  # 验证分层切分保持原始类别比例。
print([len(train), len(valid), len(test)], ratios)  # 输出集合大小和类别比例。
# -------------------- 其他写法（以下代码作为知识补充，不会执行） --------------------
# 写法A：scikit-learn可以连续两次分层切分，生产代码更常用。
# from sklearn.model_selection import train_test_split  # 导入官方切分工具。
# train_idx, temp_idx = train_test_split(np.arange(100), test_size=0.4, stratify=labels, random_state=42)  # 先切训练集。
# valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)  # 再平分临时集。
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
