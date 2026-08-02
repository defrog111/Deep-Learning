"""
题目 002：数组创建dtype与shape_变式

要求：完成“数组创建dtype与shape”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建可修改的原始字节缓冲区。
2. frombuffer零复制地查看已有内存。
3. 从迭代器创建一维数组。
4. asarray接收已有ndarray时通常直接复用而不复制。
5. 修改源缓冲区以观察共享内存效果。

完成标准：
- 验证共享视图和迭代器构造。
- 验证dtype和布局兼容时asarray复用原数组。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
source = bytearray([1, 2, 3, 4])  # 创建可修改的原始字节缓冲区。
buffer_view = np.frombuffer(source, dtype=np.uint8)  # frombuffer零复制地查看已有内存。
from_iterator = np.fromiter((value * value for value in range(5)), dtype=np.int64, count=5)  # 从迭代器创建一维数组。
existing = np.asarray(from_iterator)  # asarray接收已有ndarray时通常直接复用而不复制。
source[0] = 99  # 修改源缓冲区以观察共享内存效果。
assert buffer_view[0] == 99 and from_iterator.tolist() == [0, 1, 4, 9, 16]  # 验证共享视图和迭代器构造。
assert existing is from_iterator  # 验证dtype和布局兼容时asarray复用原数组。
print(buffer_view, from_iterator, existing)  # 输出三种不同来源的构造结果。
