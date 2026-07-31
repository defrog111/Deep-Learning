"""
题目 065：浅拷贝与深拷贝_基础

要求：完成“浅拷贝与深拷贝”的基础题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

import copy  # 导入复制工具。
original = {'items': [[1, 2], [3, 4]]}  # 创建嵌套可变对象。
shallow = copy.copy(original)  # 浅拷贝只复制最外层容器。
deep = copy.deepcopy(original)  # 深拷贝递归复制内部对象。
shallow['items'][0].append(99)  # 修改共享的内部列表会影响original。
deep['items'][1].append(88)  # 修改深拷贝内部列表不会影响original。
assert 99 in original['items'][0] and 88 not in original['items'][1]  # 验证复制深度。
print(original, shallow, deep)  # 输出三份对象。
