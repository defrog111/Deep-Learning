"""
题目 010：ColumnTransformer混合特征_变式

要求：完成“ColumnTransformer混合特征”的综合题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
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
