"""
题目 096：类型注解与Protocol_综合

要求：完成“类型注解与Protocol”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

from typing import Protocol, TypeVar  # 导入结构化类型工具。
class SupportsLen(Protocol):  # 定义只要求__len__的协议。
    def __len__(self) -> int:  # 声明所需方法签名。
        ...  # 协议无需实现。
T = TypeVar('T')  # 定义泛型类型变量。
def first(items: list[T]) -> T:  # 泛型函数保持输入元素类型。
    if not items:  # 检查空列表。
        raise ValueError('empty')  # 空输入没有首元素。
    return items[0]  # 返回同类型元素。
def length(value: SupportsLen) -> int:  # 接受任何结构上支持len的对象。
    return len(value)  # 调用协议方法。
answer = first([6, 4, 5])  # 类型检查器可推导answer为int。
assert answer == 6 and length('abc') == 3  # 验证泛型与鸭子类型。
print(answer, length([1, 2]))  # 输出类型协议结果。
