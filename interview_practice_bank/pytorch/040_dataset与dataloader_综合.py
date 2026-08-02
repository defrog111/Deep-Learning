"""
题目 040：Dataset与DataLoader_综合

要求：完成“Dataset与DataLoader”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入数据集和批加载器。
3. 导入分布式数据采样器。
4. 无需初始化进程组即可演示数据分片。
5. 验证rank1取得互不重复的奇数索引。

完成标准：
- 验证rank1取得互不重复的奇数索引。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据集和批加载器。
from torch.utils.data import DistributedSampler  # 导入分布式数据采样器。
distributed = DistributedSampler(TensorDataset(torch.arange(8)), num_replicas=2, rank=1, shuffle=False); distributed_indices = list(distributed)  # 无需初始化进程组即可演示数据分片。
assert distributed_indices == [1, 3, 5, 7]  # 验证rank1取得互不重复的奇数索引。
