"""
题目 074：Path文件CSV与JSON_变式

要求：完成“Path文件CSV与JSON”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入标准库CSV工具。
2. 导入JSON工具。
3. 导入临时目录工具。
4. 导入路径工具。
5. 导入内存文本流和CSV模块。
6. 按列名安全写CSV。
7. 读取CSV时默认字段为字符串。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import csv  # 导入标准库CSV工具。
import json  # 导入JSON工具。
import tempfile  # 导入临时目录工具。
from pathlib import Path  # 导入路径工具。
import io, csv  # 导入内存文本流和CSV模块。
buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=['name', 'score']); writer.writeheader(); writer.writerow({'name': 'A', 'score': 90})  # 按列名安全写CSV。
buffer.seek(0); csv_rows = list(csv.DictReader(buffer)); assert csv_rows[0]['score'] == '90'  # 读取CSV时默认字段为字符串。
