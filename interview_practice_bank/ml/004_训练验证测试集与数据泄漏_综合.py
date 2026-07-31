"""
题目 004：训练验证测试集与数据泄漏_综合

要求：完成“训练验证测试集与数据泄漏”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 固定随机性。
2. 构造一维回归特征。
3. 构造带噪声二次目标。
4. 生成无重复随机索引。
5. 按60/20/20切分。
6. 设置欠拟合、合适和偏复杂的候选模型。
7. 保存每个候选模型的验证误差。
8. 只使用训练集拟合并用验证集选择模型。
9. 在训练集拟合多项式。
10. 在验证集计算预测。

完成标准：
- 验证模型选择和测试结果有效。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 固定随机性。
features = np.linspace(-3, 3, 120)  # 构造一维回归特征。
targets = 1.5 * features**2 + 2 * features + rng.normal(0, 0.8, 120)  # 构造带噪声二次目标。
indices = rng.permutation(len(features))  # 生成无重复随机索引。
train, valid, test = indices[:72], indices[72:96], indices[96:]  # 按60/20/20切分。
candidate_degrees = [1, 2, 5]  # 设置欠拟合、合适和偏复杂的候选模型。
validation_mse = []  # 保存每个候选模型的验证误差。
for degree in candidate_degrees:  # 只使用训练集拟合并用验证集选择模型。
    coefficients = np.polyfit(features[train], targets[train], degree)  # 在训练集拟合多项式。
    predictions = np.polyval(coefficients, features[valid])  # 在验证集计算预测。
    validation_mse.append(np.mean((predictions - targets[valid]) ** 2))  # 保存验证MSE。
best_degree = candidate_degrees[int(np.argmin(validation_mse))]  # 根据验证误差选择复杂度。
final_coefficients = np.polyfit(features[np.r_[train, valid]], targets[np.r_[train, valid]], best_degree)  # 用训练加验证数据重训最终模型。
test_predictions = np.polyval(final_coefficients, features[test])  # 最终只评估一次测试集。
test_mse = np.mean((test_predictions - targets[test]) ** 2)  # 计算最终泛化误差。
assert best_degree in candidate_degrees and np.isfinite(test_mse)  # 验证模型选择和测试结果有效。
print(validation_mse, best_degree, test_mse)  # 输出完整模型选择结果。
# -------------------- 其他写法（以下代码作为知识补充，不会执行） --------------------
# 写法A：小数据集可用K折交叉验证代替单一验证集，让模型选择更稳定。
# from sklearn.model_selection import KFold, cross_val_score  # 导入交叉验证工具。
# 写法B：使用Pipeline把PolynomialFeatures和LinearRegression封装，避免预处理泄漏。
# from sklearn.pipeline import make_pipeline  # 导入流水线。
# from sklearn.preprocessing import PolynomialFeatures  # 导入多项式特征。
# from sklearn.linear_model import LinearRegression  # 导入线性回归。
# model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())  # 创建可交叉验证的完整流水线。
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
