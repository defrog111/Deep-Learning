"""
题目 040：Dataset与DataLoader_综合

要求：完成“Dataset与DataLoader”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建十个四维样本。
2. 创建二分类标签。
3. 把多个Tensor按第一维组成数据集。
4. 创建可复现的随机batch。
5. 取得第一个batch。

完成标准：
- 验证样本数和batch大小。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据集和批加载器。
features = torch.arange(40, dtype=torch.float32).reshape(10, 4)  # 创建十个四维样本。
labels = torch.arange(10) % 2  # 创建二分类标签。
dataset = TensorDataset(features, labels)  # 把多个Tensor按第一维组成数据集。
loader = DataLoader(dataset, batch_size=5, shuffle=True, generator=torch.Generator().manual_seed(0))  # 创建可复现的随机batch。
batch_features, batch_labels = next(iter(loader))  # 取得第一个batch。
assert len(dataset) == 10 and len(batch_features) <= 5  # 验证样本数和batch大小。
print(batch_features.shape, batch_labels)  # 输出batch。
