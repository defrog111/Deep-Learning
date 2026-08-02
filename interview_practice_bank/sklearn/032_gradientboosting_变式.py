"""
题目 032：GradientBoosting_变式

要求：完成“GradientBoosting”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 原生处理NaN并可早停。

完成标准：
- 验证缺失值推理和迭代上限。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升树。
from sklearn.model_selection import train_test_split  # 导入数据切分。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.ensemble import HistGradientBoostingClassifier  # 导入支持缺失值和直方图加速的梯度提升。
missing_features = features.astype(float).copy(); missing_features[::10, 0] = np.nan; histogram_boost = HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, early_stopping=True, random_state=42).fit(missing_features, labels)  # 原生处理NaN并可早停。
assert histogram_boost.predict(missing_features[:5]).shape == (5,) and histogram_boost.n_iter_ <= 50  # 验证缺失值推理和迭代上限。
