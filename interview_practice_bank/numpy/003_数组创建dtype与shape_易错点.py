"""
题目 003：数组创建dtype与shape_易错点

要求：完成“数组创建dtype与shape”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 混合整数、浮点和布尔会推导为共同浮点dtype。
2. astype转整数向零截断而不是四舍五入。
3. 转无符号整数会按模范围得到255。
4. 预先记录不规则嵌套列表是否被拒绝。
5. 尝试直接创建行长度不同的二维数值数组。
6. 新版NumPy会拒绝不规则shape。
7. 捕获预期的不规则数组错误。
8. 标记已经识别ragged array陷阱。

完成标准：
- 验证类型提升和截断规则。
- 验证无符号转换与不规则输入检查。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
inferred = np.array([1, 2.5, True])  # 混合整数、浮点和布尔会推导为共同浮点dtype。
truncated = np.array([1.9, -2.9]).astype(np.int64)  # astype转整数向零截断而不是四舍五入。
unsigned = np.array([-1], dtype=np.int16).astype(np.uint8)  # 转无符号整数会按模范围得到255。
ragged_failed = False  # 预先记录不规则嵌套列表是否被拒绝。
try:  # 尝试直接创建行长度不同的二维数值数组。
    np.array([[1, 2], [3]])  # 新版NumPy会拒绝不规则shape。
except ValueError:  # 捕获预期的不规则数组错误。
    ragged_failed = True  # 标记已经识别ragged array陷阱。
assert inferred.dtype == np.float64 and truncated.tolist() == [1, -2]  # 验证类型提升和截断规则。
assert unsigned.item() == 255 and ragged_failed  # 验证无符号转换与不规则输入检查。
print(inferred.dtype, truncated, unsigned)  # 输出三个dtype易错结果。
