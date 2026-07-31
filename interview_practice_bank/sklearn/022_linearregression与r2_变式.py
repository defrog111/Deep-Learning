"""
题目 022：LinearRegression与R2_变式

要求：完成“LinearRegression与R2”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 固定随机切分。
3. 拟合普通最小二乘。
4. 对未见测试集预测。
5. 计算MSE、MAE、R²。
6. 综合题统一导入NumPy用于shape、数值和标签检查。
7. 比较目标变换及L2、L1、ElasticNet，并选用稳定求解器。

完成标准：
- 验证预测shape。
- 验证四个模型均学习到有限系数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_diabetes  # 导入回归数据。
from sklearn.linear_model import LinearRegression  # 导入线性回归。
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 导入回归指标。
from sklearn.model_selection import train_test_split  # 导入切分函数。
features, targets = load_diabetes(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)  # 固定随机切分。
model = LinearRegression().fit(train_x, train_y)  # 拟合普通最小二乘。
predictions = model.predict(test_x)  # 对未见测试集预测。
metrics = mean_squared_error(test_y, predictions), mean_absolute_error(test_y, predictions), r2_score(test_y, predictions)  # 计算MSE、MAE、R²。
assert predictions.shape == test_y.shape  # 验证预测shape。
print(metrics)  # 输出回归指标。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.compose import TransformedTargetRegressor  # 导入目标变量转换包装器。
from sklearn.linear_model import Ridge, Lasso, ElasticNet  # 导入三种正则线性模型。
from sklearn.preprocessing import FunctionTransformer  # 导入显式目标函数转换器。
stable_features = features.astype(np.float32); stable_targets = targets.astype(np.float32); positive_targets = np.exp(np.linspace(0, 3, len(stable_features), dtype=np.float32)); log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False); target_model = TransformedTargetRegressor(regressor=Ridge(alpha=1.0, solver='lsqr'), transformer=log_transformer).fit(stable_features, positive_targets); regularized_models = [estimator.fit(stable_features, stable_targets) for estimator in [Ridge(solver='lsqr'), Lasso(alpha=0.1, max_iter=5000), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)]]  # 比较目标变换及L2、L1、ElasticNet，并选用稳定求解器。
assert np.isfinite(target_model.regressor_.coef_).all() and all(np.isfinite(estimator.coef_).all() for estimator in regularized_models)  # 验证四个模型均学习到有限系数。
