"""
题目 004：StandardScaler只拟合训练集_变式

要求：完成“StandardScaler只拟合训练集”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建训练数据。
2. 创建分布不同的测试数据。
3. 只能在训练集拟合均值和方差。
4. 使用训练统计量转换训练集。
5. 使用同一统计量转换测试集。
6. 综合题统一导入NumPy用于shape、数值和标签检查。
7. 只在训练集拟合并验证可逆转换。

完成标准：
- 验证没有用测试数据污染统计量。
- 验证稳健中心与inverse_transform。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
train = np.array([[1.0], [2.0], [3.0]])  # 创建训练数据。
test = np.array([[100.0]])  # 创建分布不同的测试数据。
scaler = StandardScaler().fit(train)  # 只能在训练集拟合均值和方差。
train_scaled = scaler.transform(train)  # 使用训练统计量转换训练集。
test_scaled = scaler.transform(test)  # 使用同一统计量转换测试集。
assert np.isclose(train_scaled.mean(), 0) and test_scaled.item() > 10  # 验证没有用测试数据污染统计量。
print(scaler.mean_, scaler.scale_, test_scaled)  # 输出拟合参数和转换结果。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.preprocessing import RobustScaler  # 导入基于中位数和IQR的稳健缩放器。
outlier_train = np.array([[1.0], [2.0], [3.0], [1000.0]]); robust = RobustScaler().fit(outlier_train); robust_values = robust.transform(outlier_train); restored_values = robust.inverse_transform(robust_values)  # 只在训练集拟合并验证可逆转换。
assert np.allclose(restored_values, outlier_train) and np.median(robust_values) == 0  # 验证稳健中心与inverse_transform。
