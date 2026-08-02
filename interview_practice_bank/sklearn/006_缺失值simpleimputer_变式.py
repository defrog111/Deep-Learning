"""
题目 006：缺失值SimpleImputer_变式

要求：完成“缺失值SimpleImputer”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入 NumPy。
2. 导入缺失值填充器。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 导入邻居填充和缺失指示器。
5. 同时保留缺失模式作为特征。
6. 验证填充及指示列。

完成标准：
- 验证填充及指示列。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.impute import SimpleImputer  # 导入缺失值填充器。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.impute import KNNImputer, MissingIndicator  # 导入邻居填充和缺失指示器。
complex_train = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]]); knn_filled = KNNImputer(n_neighbors=2).fit_transform(complex_train); indicators = MissingIndicator(features='all').fit_transform(complex_train)  # 同时保留缺失模式作为特征。
assert np.isfinite(knn_filled).all() and indicators.shape == complex_train.shape and indicators.sum() == 1  # 验证填充及指示列。
