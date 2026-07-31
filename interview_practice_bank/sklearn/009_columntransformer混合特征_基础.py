"""
题目 009：ColumnTransformer混合特征_基础

要求：完成“ColumnTransformer混合特征”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建混合特征表。
2. 为不同列配置不同处理。
3. 一次拟合并转换训练数据。
4. 获取转换后的特征名。
5. 两个数值列加两个类别列。

完成标准：
- 两个数值列加两个类别列。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
from sklearn.compose import ColumnTransformer  # 导入列级转换器。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入数值和类别转换器。
frame = pd.DataFrame({'age': [20, 30, 40], 'income': [30, 60, 90], 'city': ['A', 'B', 'A']})  # 创建混合特征表。
preprocessor = ColumnTransformer([('numeric', StandardScaler(), ['age', 'income']), ('category', OneHotEncoder(sparse_output=False), ['city'])])  # 为不同列配置不同处理。
transformed = preprocessor.fit_transform(frame)  # 一次拟合并转换训练数据。
names = preprocessor.get_feature_names_out()  # 获取转换后的特征名。
assert transformed.shape == (3, 4)  # 两个数值列加两个类别列。
print(names, transformed)  # 输出特征名和矩阵。
