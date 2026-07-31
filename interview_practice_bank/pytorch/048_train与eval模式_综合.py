"""
题目 048：train与eval模式_综合

要求：完成“train与eval模式”的综合题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定随机种子。
model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))  # 组合受模式影响的层。
x = torch.randn(8, 4)  # 创建batch输入。
model.train()  # 开启Dropout并更新BatchNorm运行统计。
train_output = model(x)  # 训练态前向。
model.eval()  # 关闭Dropout并冻结BatchNorm运行统计。
with torch.no_grad():  # 推理时关闭梯度记录。
    eval_output_1 = model(x)  # 第一次推理。
    eval_output_2 = model(x)  # 第二次推理。
assert torch.allclose(eval_output_1, eval_output_2)  # 验证eval输出确定性。
print(torch.allclose(train_output, eval_output_1))  # 比较训练态与推理态。
