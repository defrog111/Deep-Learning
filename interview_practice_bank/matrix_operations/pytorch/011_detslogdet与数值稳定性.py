"""
题目 011：PyTorch detslogdet与数值稳定性

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算 determinant。
2. 用 slogdet 表示符号与绝对值对数。
3. 验证二者关系并说明大矩阵优先 slogdet。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
matrix = torch.diag(torch.tensor([1e-10, 1e10, 3.0]))  # 创建尺度差异很大的可逆对角矩阵。
determinant = torch.linalg.det(matrix)  # 直接计算行列式。
sign, log_abs_det = torch.linalg.slogdet(matrix)  # 稳定返回符号和 log(abs(det))。
recovered = sign * log_abs_det.exp()  # 从对数表示恢复普通行列式。
assert torch.allclose(recovered, determinant) and torch.allclose(determinant, torch.tensor(3.0))  # 核对结果。
print(determinant, sign, log_abs_det)  # 输出普通与对数行列式。
