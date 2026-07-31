"""
题目 073：Path文件CSV与JSON_基础

要求：完成“Path文件CSV与JSON”的基础题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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
