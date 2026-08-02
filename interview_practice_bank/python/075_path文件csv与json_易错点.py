"""
题目 075：Path文件CSV与JSON_易错点

要求：完成“Path文件CSV与JSON”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入标准库CSV工具。
2. 导入JSON工具。
3. 导入临时目录工具。
4. 导入路径工具。
5. 导入Python对象序列化模块。
6. 明确记录pickle安全边界。
7. 把序列化安全作为完成标准。

完成标准：
- 把序列化安全作为完成标准。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import csv  # 导入标准库CSV工具。
import json  # 导入JSON工具。
import tempfile  # 导入临时目录工具。
from pathlib import Path  # 导入路径工具。
import pickle  # 导入Python对象序列化模块。
pickle_warning = 'pickle.loads可执行恶意构造，绝不能加载不可信数据'  # 明确记录pickle安全边界。
assert '不可信' in pickle_warning  # 把序列化安全作为完成标准。
