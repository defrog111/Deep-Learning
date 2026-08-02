"""
题目 066：浅拷贝与深拷贝_变式

要求：完成“浅拷贝与深拷贝”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入复制工具。
2. deepcopy能借助memo处理循环引用。
3. 验证循环结构被正确复制。

完成标准：
- 验证循环结构被正确复制。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import copy  # 导入复制工具。
import copy  # 导入复制工具。
recursive = []; recursive.append(recursive); recursive_copy = copy.deepcopy(recursive)  # deepcopy能借助memo处理循环引用。
assert recursive_copy is recursive_copy[0] and recursive_copy is not recursive  # 验证循环结构被正确复制。
