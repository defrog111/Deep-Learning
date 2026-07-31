"""
题目 024：函数参数args与kwargs_综合

要求：完成“函数参数args与kwargs”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

def describe(required, default=10, *args, keyword_only=True, **kwargs):  # 定义各种参数类型。
    return required, default, args, keyword_only, kwargs  # 返回绑定结果便于观察。
result = describe('x', 6, 1, 2, keyword_only=False, extra='value')  # 混合位置和关键字传参。
first, second, *rest = [10, 20, 30, 40]  # 扩展解包收集剩余元素。
merged = {**{'a': 1}, **{'b': 2}}  # 双星号解包合并映射。
assert result[2] == (1, 2) and rest == [30, 40]  # 验证参数收集和解包。
print(result, first, second, rest, merged)  # 输出参数绑定结果。
