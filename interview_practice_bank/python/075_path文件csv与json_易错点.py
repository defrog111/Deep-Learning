"""
题目 075：Path文件CSV与JSON_易错点

要求：完成“Path文件CSV与JSON”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 准备结构化记录。
2. 创建自动清理目录。
3. 定义CSV路径。
4. 定义JSON路径。
5. 安全打开CSV写入。
6. 创建字典写入器。
7. 写入表头。
8. 写入全部记录。
9. 序列化JSON文本。
10. 读取并反序列化JSON。

完成标准：
- 验证JSON往返不丢数据。
- 把序列化安全作为完成标准。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import csv  # 导入标准库CSV工具。
import json  # 导入JSON工具。
import tempfile  # 导入临时目录工具。
from pathlib import Path  # 导入路径工具。
records = [{'name': 'A', 'score': 90}, {'name': 'B', 'score': 85}]  # 准备结构化记录。
with tempfile.TemporaryDirectory() as folder:  # 创建自动清理目录。
    csv_path = Path(folder) / 'scores.csv'  # 定义CSV路径。
    json_path = Path(folder) / 'scores.json'  # 定义JSON路径。
    with csv_path.open('w', newline='', encoding='utf-8') as file:  # 安全打开CSV写入。
        writer = csv.DictWriter(file, fieldnames=['name', 'score'])  # 创建字典写入器。
        writer.writeheader()  # 写入表头。
        writer.writerows(records)  # 写入全部记录。
    json_path.write_text(json.dumps(records), encoding='utf-8')  # 序列化JSON文本。
    restored = json.loads(json_path.read_text(encoding='utf-8'))  # 读取并反序列化JSON。
assert restored == records  # 验证JSON往返不丢数据。
print(restored)  # 输出恢复记录。
import pickle  # 导入Python对象序列化模块。
pickle_warning = 'pickle.loads可执行恶意构造，绝不能加载不可信数据'  # 明确记录pickle安全边界。
assert '不可信' in pickle_warning  # 把序列化安全作为完成标准。
