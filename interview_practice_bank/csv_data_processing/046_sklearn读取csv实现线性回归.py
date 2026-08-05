"""
CSV数据处理练习 046：sklearn读取CSV实现线性回归

题目：读取房价CSV，使用train_test_split划分数据，通过Pipeline串联StandardScaler和LinearRegression，并评估MSE、R方及原始特征系数。

操作过程：
1. 定位房价回归CSV。
2. 从CSV读取模型输入数据。
3. 指定特征列并保留列名。
4. 使用DataFrame作为特征以保留可读列名。
5. 选择连续房价作为回归目标。
6. 可复现地划分80%训练集和20%测试集。
7. 串联特征标准化与线性回归。
8. 同时标准化目标并在预测时自动还原尺度。
9. 仅用训练数据拟合全部预处理器和回归参数。
10. 对未参与训练的测试数据执行推理。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.compose import TransformedTargetRegressor  # 导入目标值变换包装器。
from sklearn.linear_model import LinearRegression  # 导入普通最小二乘线性回归。
from sklearn.metrics import mean_squared_error, r2_score  # 导入常用回归评估指标。
from sklearn.model_selection import train_test_split  # 导入训练测试划分工具。
from sklearn.pipeline import Pipeline  # 导入防止预处理泄漏的流水线。
from sklearn.preprocessing import StandardScaler  # 导入特征和目标标准化器。
csv_path = Path(__file__).parents[1] / 'data' / 'house_prices.csv'  # 定位房价回归CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取模型输入数据。
feature_names = ['house_size_sqft', 'bedrooms', 'house_age_years', 'distance_km']  # 指定特征列并保留列名。
features = frame[feature_names]  # 使用DataFrame作为特征以保留可读列名。
targets = frame['price_thousands']  # 选择连续房价作为回归目标。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)  # 可复现地划分80%训练集和20%测试集。
feature_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])  # 串联特征标准化与线性回归。
model = TransformedTargetRegressor(regressor=feature_pipeline, transformer=StandardScaler())  # 同时标准化目标并在预测时自动还原尺度。
model.fit(train_x, train_y)  # 仅用训练数据拟合全部预处理器和回归参数。
predictions = model.predict(test_x)  # 对未参与训练的测试数据执行推理。
mse = mean_squared_error(test_y, predictions)  # 使用sklearn计算均方误差。
r2 = r2_score(test_y, predictions)  # 使用sklearn计算决定系数R方。
linear_model = model.regressor_.named_steps['regressor']  # 从已拟合流水线中取得线性回归步骤。
coefficients = pd.Series(linear_model.coef_, index=feature_names, name='scaled_coefficient')  # 给标准化空间中的系数加上特征名称。
assert len(predictions) == len(test_y) and mse >= 0.0 and r2 > 0.95  # 验证预测数量、MSE范围和拟合效果。
print('coefficients:', coefficients, 'intercept:', linear_model.intercept_, 'mse:', mse, 'r2:', r2, sep='\n')  # 输出系数、截距和测试指标。
# 简单写法：LinearRegression().fit(train_x, train_y)可直接拟合，但没有展示标准化流水线。
# 正则化替代：可把LinearRegression换成Ridge(alpha=1.0)或Lasso(alpha=0.1)控制过拟合。
# 易错点：必须先划分数据再fit流水线，不能在全部数据上提前执行StandardScaler().fit_transform。
