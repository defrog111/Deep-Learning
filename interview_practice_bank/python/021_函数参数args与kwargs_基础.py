"""
题目 021：函数参数args与kwargs_基础

要求：完成“函数参数args与kwargs”的基础题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义各种参数类型。
2. 返回绑定结果便于观察。
3. 混合位置和关键字传参。
4. 扩展解包收集剩余元素。
5. 双星号解包合并映射。

完成标准：
- 验证参数收集和解包。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def describe(required, default=10, *args, keyword_only=True, **kwargs):  # 定义各种参数类型。
    return required, default, args, keyword_only, kwargs  # 返回绑定结果便于观察。
result = describe('x', 3, 1, 2, keyword_only=False, extra='value')  # 混合位置和关键字传参。
first, second, *rest = [10, 20, 30, 40]  # 扩展解包收集剩余元素。
merged = {**{'a': 1}, **{'b': 2}}  # 双星号解包合并映射。
assert result[2] == (1, 2) and rest == [30, 40]  # 验证参数收集和解包。
print(result, first, second, rest, merged)  # 输出参数绑定结果。
