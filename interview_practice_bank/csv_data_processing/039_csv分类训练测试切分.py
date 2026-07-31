"""
CSV数据处理练习 039：CSV分类训练测试切分

题目：读取Iris CSV，编码标签、分层切分训练测试集，并验证类别比例。

操作过程：
1. 定位Iris CSV。
2. 从CSV读取分类数据。
3. 提取数值特征。
4. 把类别名称编码为0到K减1。
5. 分层切分保持类别比例。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import numpy as np  # 导入NumPy。
import pandas as pd  # 导入Pandas。
from sklearn.model_selection import train_test_split  # 导入切分函数。
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器。
csv_path = Path(__file__).parents[2] / 'transformer_learning' / 'data' / 'iris.csv'  # 定位Iris CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取分类数据。
features = frame.drop(columns='species').to_numpy(dtype=np.float32)  # 提取数值特征。
encoder = LabelEncoder(); labels = encoder.fit_transform(frame['species'])  # 把类别名称编码为0到K减1。
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)  # 分层切分保持类别比例。
assert train_x.shape == (120, 4) and test_x.shape == (30, 4) and np.array_equal(np.bincount(test_y), [10, 10, 10])  # 验证shape和测试类别数。
print(encoder.classes_, train_x.shape, test_x.shape)  # 输出标签映射和集合shape。
