"""
题目 039：Dataset与DataLoader_易错点

要求：完成“Dataset与DataLoader”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据集和批加载器。
features = torch.arange(40, dtype=torch.float32).reshape(10, 4)  # 创建十个四维样本。
labels = torch.arange(10) % 2  # 创建二分类标签。
dataset = TensorDataset(features, labels)  # 把多个Tensor按第一维组成数据集。
loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))  # 创建可复现的随机batch。
batch_features, batch_labels = next(iter(loader))  # 取得第一个batch。
assert len(dataset) == 10 and len(batch_features) <= 4  # 验证样本数和batch大小。
print(batch_features.shape, batch_labels)  # 输出batch。
