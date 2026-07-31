"""
CSV数据处理练习 038：CSV到sklearn预处理Pipeline

题目：读取客户CSV，使用ColumnTransformer分别标准化数值列和独热编码类别列。

操作过程：
1. 定位客户CSV。
2. 读取混合类型客户特征。
3. 为不同列配置转换。
4. 只在训练示例数据上拟合并转换。
5. 取得转换后列名。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import numpy as np  # 导入NumPy用于数值检查。
import pandas as pd  # 导入Pandas。
from sklearn.compose import ColumnTransformer  # 导入列转换器。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入数值和类别转换器。
csv_path = Path(__file__).parents[1] / 'data' / 'customers.csv'  # 定位客户CSV。
frame = pd.read_csv(csv_path)  # 读取混合类型客户特征。
preprocessor = ColumnTransformer([('numeric', StandardScaler(), ['age']), ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['segment', 'city'])])  # 为不同列配置转换。
matrix = preprocessor.fit_transform(frame)  # 只在训练示例数据上拟合并转换。
feature_names = preprocessor.get_feature_names_out()  # 取得转换后列名。
assert matrix.shape[0] == len(frame) and np.isfinite(matrix).all() and len(feature_names) == matrix.shape[1]  # 验证样本数、数值和特征名。
print(feature_names, matrix[:2], sep='\n')  # 输出特征空间。
