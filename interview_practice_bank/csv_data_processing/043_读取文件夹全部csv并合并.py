"""
CSV数据处理练习 043：读取文件夹全部CSV并合并

题目：扫描data文件夹中的全部CSV，逐个读取和清洗列名，记录来源与文件质量，再兼容不同列结构完成纵向合并和分组核对。

操作过程：
1. 定位存放多个CSV文件的共享数据目录。
2. 排除可能由本脚本生成的汇总文件，避免下次运行重复读入。
3. 查找当前文件夹的全部CSV并固定读取顺序。
4. 在循环前检查文件列表，避免pd.concat收到空列表。
5. 用明确异常说明缺少输入文件。
6. 准备保存每个CSV对应的DataFrame。
7. 准备保存每个文件的行数、列数和缺失值统计。
8. 逐个处理文件夹中的CSV文件。
9. 从磁盘读取当前CSV文件。
10. 统一清理列名以便后续合并和查询。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
data_folder = Path(__file__).parents[1] / 'data'  # 定位存放多个CSV文件的共享数据目录。
excluded_names = {'all_csv_combined.csv'}  # 排除可能由本脚本生成的汇总文件，避免下次运行重复读入。
csv_paths = sorted(path for path in data_folder.glob('*.csv') if path.name not in excluded_names)  # 查找当前文件夹的全部CSV并固定读取顺序。
if not csv_paths:  # 在循环前检查文件列表，避免pd.concat收到空列表。
    raise FileNotFoundError(f'文件夹中没有可读取的CSV: {data_folder}')  # 用明确异常说明缺少输入文件。
frames = []  # 准备保存每个CSV对应的DataFrame。
file_audits = []  # 准备保存每个文件的行数、列数和缺失值统计。
for csv_path in csv_paths:  # 逐个处理文件夹中的CSV文件。
    current = pd.read_csv(csv_path)  # 从磁盘读取当前CSV文件。
    current.columns = current.columns.str.strip().str.lower().str.replace(' ', '_')  # 统一清理列名以便后续合并和查询。
    if 'source_file' in current.columns:  # 防止输入文件已有同名列而被静默覆盖。
        raise ValueError(f'{csv_path.name} 已包含保留列source_file')  # 明确报告来源列冲突。
    original_column_count = current.shape[1]  # 在添加来源列之前记录业务列数量。
    current.insert(0, 'source_file', csv_path.name)  # 添加来源文件名以保留数据血缘。
    file_audits.append({'source_file': csv_path.name, 'rows': len(current), 'columns': original_column_count, 'missing_cells': int(current.isna().sum().sum())})  # 汇总当前文件的质量指标。
    frames.append(current)  # 把处理后的表加入待合并列表。
combined = pd.concat(frames, ignore_index=True, join='outer', sort=False)  # 用outer保留不同CSV的全部列并重建连续索引。
audit = pd.DataFrame(file_audits).sort_values('source_file').reset_index(drop=True)  # 构造按文件名排序的读取审计表。
rows_by_source = combined.groupby('source_file', sort=True).size().rename('combined_rows')  # 合并后按来源重新统计行数。
audit = audit.merge(rows_by_source, left_on='source_file', right_index=True, validate='one_to_one')  # 一对一合并前后统计以便核对。
selected = combined.loc[combined['source_file'].eq(csv_paths[0].name)].copy()  # 演示按来源文件筛选合并结果。
assert set(audit['source_file']) == {path.name for path in csv_paths}  # 验证扫描到的每个CSV都完成了读取。
assert len(combined) == audit['rows'].sum() and audit['rows'].eq(audit['combined_rows']).all()  # 验证合并总行数及逐文件行数均未改变。
assert len(selected) == int(audit.loc[audit['source_file'].eq(csv_paths[0].name), 'rows'].iloc[0])  # 验证按来源筛选得到正确行数。
print('读取文件:', [path.name for path in csv_paths])  # 输出实际读取的全部文件名。
print('文件审计:\n', audit)  # 输出每个CSV的结构和缺失值摘要。
print('合并形状:', combined.shape, '全部列:', combined.columns.tolist())  # 输出outer合并后的形状及列并集。
print('合并预览:\n', combined.head())  # 输出带来源信息的合并结果预览。
# 其他写法：data_folder.rglob('*.csv')可以递归读取当前目录及所有子目录。
# 其他写法：glob.glob(str(data_folder / '*.csv'))返回字符串路径，读取前可再包装成Path。
# 同结构文件可使用pd.concat(frames, ignore_index=True, join='inner')只保留所有文件共有的列。
# 易错点：不要把输出CSV写回扫描目录后直接用glob重读，否则输出会在下一次运行中成为输入。
