"""
CSV数据处理练习 040：CSV到PyTorch Dataset

题目：读取Iris CSV，把特征和标签转换为TensorDataset，再用DataLoader产生batch。

操作过程：
1. 定位Iris CSV。
2. 从CSV读取数据。
3. 转换数值特征为float32张量。
4. CrossEntropyLoss标签使用long。
5. 创建可复现随机batch。
6. 取得第一个batch。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
import torch  # 导入PyTorch。
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器。
from torch.utils.data import DataLoader, TensorDataset  # 导入张量数据集和加载器。
csv_path = Path(__file__).parents[2] / 'transformer_learning' / 'data' / 'iris.csv'  # 定位Iris CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取数据。
feature_tensor = torch.tensor(frame.drop(columns='species').to_numpy(), dtype=torch.float32)  # 转换数值特征为float32张量。
label_tensor = torch.tensor(LabelEncoder().fit_transform(frame['species']), dtype=torch.long)  # CrossEntropyLoss标签使用long。
loader = DataLoader(TensorDataset(feature_tensor, label_tensor), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(42))  # 创建可复现随机batch。
batch_features, batch_labels = next(iter(loader))  # 取得第一个batch。
assert batch_features.shape == (16, 4) and batch_labels.dtype == torch.long  # 验证batch shape和标签dtype。
print(batch_features.shape, batch_labels[:5])  # 输出batch信息。
