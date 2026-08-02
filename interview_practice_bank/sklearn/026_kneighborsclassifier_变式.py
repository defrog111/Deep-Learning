"""
题目 026：KNeighborsClassifier_变式

要求：完成“KNeighborsClassifier”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. KNN对尺度敏感需标准化。
2. 加载数据。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 距离加权且为无邻居样本提供兜底标签。

完成标准：
- 验证半径近邻推理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.model_selection import cross_val_score  # 导入交叉验证。
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # KNN对尺度敏感需标准化。
features, labels = load_iris(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.neighbors import RadiusNeighborsClassifier  # 导入固定半径近邻分类器。
radius_model = make_pipeline(StandardScaler(), RadiusNeighborsClassifier(radius=2.0, weights='distance', outlier_label='most_frequent')).fit(features, labels); radius_predictions = radius_model.predict(features[:5])  # 距离加权且为无邻居样本提供兜底标签。
assert radius_predictions.shape == (5,)  # 验证半径近邻推理。
