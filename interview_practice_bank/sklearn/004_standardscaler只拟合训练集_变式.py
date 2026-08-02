"""
题目 004：StandardScaler只拟合训练集_变式

要求：完成“StandardScaler只拟合训练集”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入 NumPy。
2. 导入标准化器。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 导入基于中位数和IQR的稳健缩放器。
5. 只在训练集拟合并验证可逆转换。
6. 验证稳健中心与inverse_transform。

完成标准：
- 验证稳健中心与inverse_transform。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.preprocessing import RobustScaler  # 导入基于中位数和IQR的稳健缩放器。
outlier_train = np.array([[1.0], [2.0], [3.0], [1000.0]]); robust = RobustScaler().fit(outlier_train); robust_values = robust.transform(outlier_train); restored_values = robust.inverse_transform(robust_values)  # 只在训练集拟合并验证可逆转换。
assert np.allclose(restored_values, outlier_train) and np.median(robust_values) == 0  # 验证稳健中心与inverse_transform。
