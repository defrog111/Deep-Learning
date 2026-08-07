"""
CSV数据处理练习 045：PyTorch读取CSV训练线性回归

题目：读取房价CSV并转换为Tensor，划分和标准化训练测试集，使用nn.Linear与Adam训练多元线性回归，再在独立推理阶段计算MSE和R方。

操作过程：
1. 定位房价回归数据文件。
2. 从CSV读取完整回归数据。
3. 定义模型输入列。
4. 把CSV特征转换为二维浮点Tensor。
5. 把连续目标转换为shape=(N,1)。
6. 使用train_test_split同时划分训练和测试特征及目标。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
from sklearn.model_selection import train_test_split  # 导入训练测试集切分工具。
csv_path = Path(__file__).parents[1] / 'data' / 'house_prices.csv'  # 定位房价回归数据文件。
frame = pd.read_csv(csv_path)  # 从CSV读取完整回归数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
feature_names = ['house_size_sqft', 'bedrooms', 'house_age_years', 'distance_km']  # 定义模型输入列。
# drop变体：feature_frame = frame.drop(columns=['price_thousands'])  # 按列名排除目标列得到全部特征。
# iloc变体：feature_frame = frame.iloc[:, :-1]  # 按位置选择目标列之前的全部特征。
features = torch.tensor(frame[feature_names].to_numpy(), dtype=torch.float32)  # 把CSV特征转换为二维浮点Tensor。
targets = torch.tensor(frame['price_thousands'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 把连续目标转换为shape=(N,1)。
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)  # 随机且可复现地划分80%训练集和20%测试集。
# 原来的PyTorch手动切分写法：先生成随机排列，再用同一组索引切分特征和目标，保证样本一一对应。
# generator = torch.Generator().manual_seed(42)
# indices = torch.randperm(len(frame), generator=generator)
# test_indices, train_indices = indices[:6], indices[6:]
# train_x, test_x = features[train_indices], features[test_indices]
# train_y, test_y = targets[train_indices], targets[test_indices]
x_mean, x_std = train_x.mean(dim=0, keepdim=True), train_x.std(dim=0, keepdim=True)  # 只从训练特征计算标准化参数。
y_mean, y_std = train_y.mean(dim=0, keepdim=True), train_y.std(dim=0, keepdim=True)  # 标准化目标以加快梯度训练收敛。
train_x_scaled = (train_x - x_mean) / x_std  # 标准化训练特征。
test_x_scaled = (test_x - x_mean) / x_std  # 使用训练统计量标准化测试特征。
train_y_scaled = (train_y - y_mean) / y_std  # 标准化训练目标。
# StandardScaler等价写法：只在训练集fit，测试集只能使用同一个scaler执行transform，避免数据泄漏。
# from sklearn.preprocessing import StandardScaler
# x_scaler = StandardScaler()
# train_x_scaled = torch.tensor(x_scaler.fit_transform(train_x.numpy()), dtype=torch.float32)
# test_x_scaled = torch.tensor(x_scaler.transform(test_x.numpy()), dtype=torch.float32)
torch.manual_seed(42)  # 固定模型参数初始化以保证结果可复现。
model = nn.Linear(len(feature_names), 1)  # 创建包含权重和偏置的单层线性回归模型。
loss_function = nn.MSELoss()  # 使用均方误差作为回归损失。
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 创建Adam优化器更新模型参数。
for epoch in range(800):  # 多轮使用全部训练样本进行梯度下降。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清除上一轮累积的参数梯度。
    train_predictions = model(train_x_scaled)  # 前向传播得到标准化空间中的训练预测。
    loss = loss_function(train_predictions, train_y_scaled)  # 计算本轮训练损失。
    loss.backward()  # 通过自动微分计算参数梯度。
    optimizer.step()  # 根据梯度更新权重和偏置。
model.eval()  # 训练结束后切换到推理模式。
with torch.no_grad():  # 推理阶段关闭梯度跟踪以节省内存。
    scaled_predictions = model(test_x_scaled)  # 对测试特征执行独立推理。
    predictions = scaled_predictions * y_std + y_mean  # 把预测值还原到原始房价尺度。
    mse = torch.mean((test_y - predictions) ** 2)  # 在原始尺度计算测试均方误差。
    r2 = 1.0 - torch.sum((test_y - predictions) ** 2) / torch.sum((test_y - test_y.mean()) ** 2)  # 计算测试集R方。
assert predictions.shape == test_y.shape and torch.isfinite(mse) and r2.item() > 0.95  # 验证输出形状、有限指标和拟合效果。
print('weight:', model.weight.detach(), 'bias:', model.bias.detach(), 'mse:', mse.item(), 'r2:', r2.item(), sep='\n')  # 输出训练参数与测试指标。
# 替代写法：torch.optim.SGD(model.parameters(), lr=0.01)可使用随机梯度下降优化器。
# 易错点：targets若保持shape=(N,)，会和shape=(N,1)的预测发生广播，应使用unsqueeze(1)。
# 易错点：model.eval()不会自动关闭梯度，推理时还应配合torch.no_grad()或torch.inference_mode()。
