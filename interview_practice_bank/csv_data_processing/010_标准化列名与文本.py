"""
CSV数据处理练习 010：标准化列名与文本

题目：读取脏订单CSV，把状态和渠道统一为去空格的小写形式。

操作过程：
1. 定位脏订单CSV。
2. 读取脏文本数据。
3. 统一列名风格。
4. 向量化清洗状态。
5. 统一渠道首字母大写。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'orders_dirty.csv'  # 定位脏订单CSV。
frame = pd.read_csv(csv_path)  # 读取脏文本数据。
frame.columns = frame.columns.str.strip().str.lower().str.replace(' ', '_')  # 统一列名风格。
frame['status'] = frame['status'].astype('string').str.strip().str.lower()  # 向量化清洗状态。
frame['channel'] = frame['channel'].astype('string').str.strip().str.title()  # 统一渠道首字母大写。
assert set(frame['status']) <= {'completed', 'pending', 'cancelled', 'returned'}  # 验证状态词汇已归一化。
print(frame[['status', 'channel']].drop_duplicates())  # 输出清洗后的类别组合。
