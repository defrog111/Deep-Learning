"""
CSV数据处理练习 046：sklearn读取CSV实现线性回归

题目：读取房价CSV，使用train_test_split划分数据，直接用LinearRegression拟合，并评估MSE、R方及各特征系数。

操作过程：
1. 定位房价回归CSV。
2. 从CSV读取模型输入数据。
3. 指定特征列并保留列名。
4. 使用DataFrame作为特征以保留可读列名。
5. 选择连续房价作为回归目标。
6. 可复现地划分80%训练集和20%测试集。
7. 创建普通最小二乘线性回归模型。
8. 仅用训练数据拟合回归参数。
9. 对未参与训练的测试数据执行推理。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.linear_model import LinearRegression  # 导入普通最小二乘线性回归。
from sklearn.metrics import mean_squared_error, r2_score  # 导入常用回归评估指标。
from sklearn.model_selection import train_test_split  # 导入训练测试划分工具。
csv_path = Path(__file__).parents[1] / 'data' / 'house_prices.csv'  # 定位房价回归CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取模型输入数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
feature_names = ['house_size_sqft', 'bedrooms', 'house_age_years', 'distance_km']  # 指定特征列并保留列名。
# drop变体：features = frame.drop(columns=['price_thousands'])  # 按列名排除目标列得到全部特征。
# iloc变体：features = frame.iloc[:, :-1]  # 按位置选择目标列之前的全部特征。
features = frame[feature_names]  # 使用DataFrame作为特征以保留可读列名。
targets = frame['price_thousands']  # 选择连续房价作为回归目标。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)  # 可复现地划分80%训练集和20%测试集。
model = LinearRegression()  # 创建自动学习特征系数和截距的普通最小二乘模型。
model.fit(train_x, train_y)  # 仅用训练数据拟合回归参数。
predictions = model.predict(test_x)  # 对未参与训练的测试数据执行推理。
mse = mean_squared_error(test_y, predictions)  # 使用sklearn计算均方误差。
r2 = r2_score(test_y, predictions)  # 使用sklearn计算决定系数R方。
coefficients = pd.Series(model.coef_, index=feature_names, name='coefficient')  # 给原始特征尺度下的系数加上特征名称。
assert len(predictions) == len(test_y) and mse >= 0.0 and r2 > 0.95  # 验证预测数量、MSE范围和拟合效果。
print('coefficients:', coefficients, 'intercept:', model.intercept_, 'mse:', mse, 'r2:', r2, sep='\n')  # 输出系数、截距和测试指标。
# 正则化替代：可把LinearRegression换成Ridge(alpha=1.0)或Lasso(alpha=0.1)控制过拟合。
# 提示：LinearRegression默认fit_intercept=True，会自动学习截距，不需要手动添加全1列。
