"""
题目 038：Dataset与DataLoader_变式

要求：完成“Dataset与DataLoader”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入数据集和批加载器。
3. 导入自定义数据集基类。
class PairDataset(Dataset):  # 定义最小地图式数据集。
    def __len__(self):  # 返回样本总数。
        return 5  # 固定五个样本。
    def __getitem__(self, index):  # 按索引生成样本。
        return torch.tensor(index), torch.tensor(index % 2)  # 返回特征标签对。
4. 验证Dataset与默认collate。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据集和批加载器。
from torch.utils.data import Dataset  # 导入自定义数据集基类。
class PairDataset(Dataset):  # 定义最小地图式数据集。
    def __len__(self):  # 返回样本总数。
        return 5  # 固定五个样本。
    def __getitem__(self, index):  # 按索引生成样本。
        return torch.tensor(index), torch.tensor(index % 2)  # 返回特征标签对。
custom_batch = next(iter(DataLoader(PairDataset(), batch_size=3))); assert custom_batch[0].shape == (3,)  # 验证Dataset与默认collate。
