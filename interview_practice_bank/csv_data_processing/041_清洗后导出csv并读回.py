"""
CSV数据处理练习 041：清洗后导出CSV并读回

题目：读取脏订单CSV，清洗金额、状态和重复值，写入临时CSV后再次读回验证。

操作过程：
1. 定位脏订单CSV。
2. 读取并建立独立清洗表。
3. 清洗状态。
4. 清洗金额。
5. 创建自动删除的临时目录。
6. 定义输出CSV。
7. 不写DataFrame索引。
8. 从导出CSV重新读取。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入自动清理临时目录。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path).drop_duplicates('order_id').copy()  # 读取并建立独立清洗表。
frame['status'] = frame['status'].str.strip().str.lower()  # 清洗状态。
frame['amount'] = pd.to_numeric(frame['amount'].astype('string').str.replace(r'[$,]', '', regex=True), errors='coerce')  # 清洗金额。
with tempfile.TemporaryDirectory() as folder:  # 创建自动删除的临时目录。
    output_path = Path(folder) / 'orders_clean.csv'  # 定义输出CSV。
    frame.to_csv(output_path, index=False, date_format='%Y-%m-%d')  # 不写DataFrame索引。
    restored = pd.read_csv(output_path)  # 从导出CSV重新读取。
assert restored.shape == frame.shape and restored['order_id'].is_unique  # 验证往返行列数和业务键。
print(restored.head())  # 输出恢复数据。
