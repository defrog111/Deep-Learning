"""
CSV数据处理练习 044：NumPy读取CSV实现线性回归

题目：读取房价CSV，手动划分训练测试集，仅用训练集标准化特征，并用NumPy最小二乘拟合多元线性回归，最后计算MSE和R方。

操作过程：
1. 定位房价回归数据文件。
2. 从CSV读取特征和连续目标值。
3. 指定四个输入特征。
4. 把DataFrame特征转换为浮点矩阵。
5. 把房价列转换为一维目标数组。
6. 使用train_test_split同时划分训练和测试特征及目标。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import numpy as np  # 导入NumPy进行矩阵运算。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.model_selection import train_test_split  # 导入训练测试集切分工具。
csv_path = Path(__file__).parents[1] / 'data' / 'house_prices.csv'  # 定位房价回归数据文件。
frame = pd.read_csv(csv_path)  # 从CSV读取特征和连续目标值。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
feature_names = ['house_size_sqft', 'bedrooms', 'house_age_years', 'distance_km']  # 指定四个输入特征。
# drop变体：feature_frame = frame.drop(columns=['price_thousands'])  # 按列名排除目标列得到全部特征。
# iloc变体：feature_frame = frame.iloc[:, :-1]  # 按位置选择目标列之前的全部特征。
features = frame[feature_names].to_numpy(dtype=np.float64)  # 把DataFrame特征转换为浮点矩阵。
targets = frame['price_thousands'].to_numpy(dtype=np.float64)  # 把房价列转换为一维目标数组。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)  # 随机且可复现地划分80%训练集和20%测试集。
# 原来的NumPy手动切分写法：先打乱索引，再用同一组索引切分特征和目标，保证样本一一对应。
# rng = np.random.default_rng(42)
# indices = rng.permutation(len(frame))
# test_indices, train_indices = indices[:6], indices[6:]
# train_x, test_x = features[train_indices], features[test_indices]
# train_y, test_y = targets[train_indices], targets[test_indices]
mean = train_x.mean(axis=0, keepdims=True)  # 只用训练集计算每列均值以避免数据泄漏。
std = train_x.std(axis=0, keepdims=True)  # 只用训练集计算每列标准差。
train_scaled = (train_x - mean) / std  # 标准化训练特征以改善数值条件。
test_scaled = (test_x - mean) / std  # 使用训练集统计量转换测试特征。
train_design = np.column_stack((np.ones(len(train_scaled)), train_scaled))  # 添加全1截距列构建设计矩阵。
test_design = np.column_stack((np.ones(len(test_scaled)), test_scaled))  # 为测试特征添加相同截距列。
weights, residuals, rank, singular_values = np.linalg.lstsq(train_design, train_y, rcond=None)  # 用稳定的最小二乘求截距和各特征系数。
predictions = test_design @ weights  # 用矩阵乘法完成测试集推理。
mse = np.mean((test_y - predictions) ** 2)  # 计算均方误差。
r2 = 1.0 - np.sum((test_y - predictions) ** 2) / np.sum((test_y - test_y.mean()) ** 2)  # 手动计算决定系数R方。
assert weights.shape == (len(feature_names) + 1,) and rank == len(feature_names) + 1  # 验证参数数量和设计矩阵满秩。
assert predictions.shape == test_y.shape and np.isfinite(mse) and r2 > 0.95  # 验证预测形状、指标有效且拟合合理。
print('weights:', weights, 'mse:', mse, 'r2:', r2, sep='\n')  # 输出模型参数和测试指标。
# 替代写法：weights = np.linalg.pinv(train_design) @ train_y可用伪逆处理不可逆或非方阵。
# 公式写法：weights = np.linalg.solve(train_design.T @ train_design, train_design.T @ train_y)，但正规方程数值稳定性通常较差。
# 训练写法：也可初始化weights后循环执行weights -= learning_rate * train_design.T @ (train_design @ weights - train_y) / len(train_y)。
# 用 np.c_
#train_design = np.c_[np.ones(len(train_scaled)), train_scaled]

# 用 np.concatenate
#train_design = np.concatenate([np.ones((len(train_scaled), 1)), train_scaled],axis=1)
