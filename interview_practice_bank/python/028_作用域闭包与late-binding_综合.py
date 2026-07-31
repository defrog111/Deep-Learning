"""
题目 028：作用域闭包与late-binding_综合

要求：完成“作用域闭包与late-binding”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

def make_multiplier(factor):  # 外层函数创建闭包环境。
    def multiply(value):  # 内层函数引用外层变量。
        return value * factor  # factor被闭包保存。
    return multiply  # 返回函数对象。
times_n = make_multiplier(6)  # 创建具体倍数闭包。
bad = [lambda: index for index in range(3)]  # late-binding导致调用时都读取最终index。
fixed = [lambda index=index: index for index in range(3)]  # 默认参数在定义时绑定当前值。
assert [fn() for fn in bad] == [2, 2, 2] and [fn() for fn in fixed] == [0, 1, 2]  # 验证闭包陷阱。
print(times_n(5), [fn() for fn in fixed])  # 输出闭包结果。
