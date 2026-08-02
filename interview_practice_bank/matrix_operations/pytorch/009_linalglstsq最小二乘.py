"""
题目 009：PyTorch linalglstsq最小二乘

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 构建设计矩阵和含噪标签。
2. 调用 torch.linalg.lstsq。
3. 检查 solution、rank、singular_values 和 residuals。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
x = torch.arange(5.0)  # 创建单特征样本。
design = torch.stack((x, torch.ones_like(x)), dim=1)  # 添加截距列得到 (5,2)。
targets = torch.tensor([1.1, 2.9, 5.2, 6.8, 9.1])  # 创建含噪线性目标。
result = torch.linalg.lstsq(design, targets)  # 求解最小化 ||Ax-b|| 的参数。
parameters = result.solution  # 取得斜率和截距。
predictions = design @ parameters  # 计算拟合值。
assert parameters.shape == (2,) and int(result.rank) == 2  # 验证返回 shape 和满列秩。
print(parameters, predictions, result.residuals, result.singular_values)  # 输出诊断信息。
