"""
题目 048：train与eval模式_综合

要求：完成“train与eval模式”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 组合受模式影响的层。
3. 创建batch输入。
4. 开启Dropout并更新BatchNorm运行统计。
5. 训练态前向。
6. 关闭Dropout并冻结BatchNorm运行统计。
7. 推理时关闭梯度记录。
8. 第一次推理。
9. 第二次推理。
10. 比较训练态与推理态。

完成标准：
- 验证eval输出确定性。
- 综合验证模式与梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
mode_model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout()); mode_model.eval()  # eval递归切换所有子模块模式。
with torch.inference_mode():  # 推理还需单独关闭梯度。
    mode_prediction = mode_model(torch.ones(1, 2))  # 执行确定性推理。
assert not mode_model.training and not mode_model[0].training and not mode_prediction.requires_grad  # 综合验证模式与梯度。
