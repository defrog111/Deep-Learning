"""
题目 010：ColumnTransformer混合特征_变式

要求：完成“ColumnTransformer混合特征”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入 Pandas。
2. 导入列级转换器。
3. 导入数值和类别转换器。
4. 综合题统一导入NumPy用于shape、数值和标签检查。
5. 导入数值缺失填充器。
6. 导入子流水线构造函数。
7. 每类列使用独立流水线。
8. 验证混合预处理无缺失。

完成标准：
- 验证混合预处理无缺失。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
from sklearn.compose import ColumnTransformer  # 导入列级转换器。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入数值和类别转换器。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.impute import SimpleImputer  # 导入数值缺失填充器。
from sklearn.pipeline import make_pipeline  # 导入子流水线构造函数。
mixed = pd.DataFrame({'age': [20.0, None, 40.0], 'income': [30.0, 60.0, 90.0], 'city': ['A', 'B', 'A']}); robust_preprocessor = ColumnTransformer([('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['age', 'income']), ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['city'])], verbose_feature_names_out=False); mixed_result = robust_preprocessor.fit_transform(mixed)  # 每类列使用独立流水线。
assert mixed_result.shape == (3, 4) and np.isfinite(mixed_result).all()  # 验证混合预处理无缺失。
