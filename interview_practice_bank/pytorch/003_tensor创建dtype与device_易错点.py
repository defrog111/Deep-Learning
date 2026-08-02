"""
题目 003：Tensor创建dtype与device_易错点

要求：完成“Tensor创建dtype与device”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建显式int32 Tensor。
2. 创建显式float64 Tensor。
3. 混合dtype按照类型提升规则得到float64。
4. 创建用于演示to非原地语义的Tensor。
5. to返回转换后的新Tensor而不修改original。
6. 记录整数Tensor开启梯度是否失败。
7. 尝试让整数Tensor记录梯度。
8. Autograd只支持浮点或复数Tensor梯度。
9. 捕获预期的dtype错误。
10. 标记已识别梯度dtype陷阱。

完成标准：
- 验证类型提升和to非原地语义。
- 验证显式转换与梯度限制。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
integer = torch.tensor([1, 2], dtype=torch.int32)  # 创建显式int32 Tensor。
floating = torch.tensor([0.5, 1.5], dtype=torch.float64)  # 创建显式float64 Tensor。
promoted = integer + floating  # 混合dtype按照类型提升规则得到float64。
original = torch.tensor([1.0, 2.0])  # 创建用于演示to非原地语义的Tensor。
converted = original.to(dtype=torch.float64)  # to返回转换后的新Tensor而不修改original。
requires_grad_failed = False  # 记录整数Tensor开启梯度是否失败。
try:  # 尝试让整数Tensor记录梯度。
    torch.tensor([1, 2], requires_grad=True)  # Autograd只支持浮点或复数Tensor梯度。
except RuntimeError:  # 捕获预期的dtype错误。
    requires_grad_failed = True  # 标记已识别梯度dtype陷阱。
assert promoted.dtype == torch.float64 and original.dtype == torch.float32  # 验证类型提升和to非原地语义。
assert converted.dtype == torch.float64 and requires_grad_failed  # 验证显式转换与梯度限制。
print(promoted, original.dtype, converted.dtype)  # 输出dtype易错结果。
