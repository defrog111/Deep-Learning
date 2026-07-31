"""
题目 021：LinearRegression与R2_基础

要求：完成“LinearRegression与R2”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 固定随机切分。
3. 拟合普通最小二乘。
4. 对未见测试集预测。
5. 计算MSE、MAE、R²。

完成标准：
- 验证预测shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_diabetes  # 导入回归数据。
from sklearn.linear_model import LinearRegression  # 导入线性回归。
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 导入回归指标。
from sklearn.model_selection import train_test_split  # 导入切分函数。
features, targets = load_diabetes(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=41)  # 固定随机切分。
model = LinearRegression().fit(train_x, train_y)  # 拟合普通最小二乘。
predictions = model.predict(test_x)  # 对未见测试集预测。
metrics = mean_squared_error(test_y, predictions), mean_absolute_error(test_y, predictions), r2_score(test_y, predictions)  # 计算MSE、MAE、R²。
assert predictions.shape == test_y.shape  # 验证预测shape。
print(metrics)  # 输出回归指标。
