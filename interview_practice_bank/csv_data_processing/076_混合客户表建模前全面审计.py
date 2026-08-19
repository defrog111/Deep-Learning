"""
CSV数据处理练习 076：混合客户表建模前全面审计

题目：在选择分类模型之前，全面研究混合客户购买CSV的字段角色、数据类型、类别取值、缺失值、分布、异常值、重复值、标签平衡和建模风险。

操作过程：
1. 读取CSV并列出全部字段名称、原始dtype和业务角色。
2. 区分数值、日期、类别、标识符、隐私文本、标签和分区字段。
3. 统计每列非空量、NA数量、NA比例、唯一值数量和唯一值比例。
4. 列出每个类别字段包含的类别及频数。
5. 汇总数值分布、日期范围、无效日期、重复行和重复客户ID。
6. 用IQR统计数值异常值并计算数值相关性。
7. 检查训练验证推理分区、标签缺失和类别不平衡。
8. 标记高基数字段、常量字段、隐私字段及潜在数据泄漏。
9. 给出建议使用的模型特征、排除字段和候选分类模型。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas进行CSV审计和统计。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位真实风格混合客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取完整原始表格。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全空记录并重建连续索引。
all_columns = frame.columns.tolist()  # 列出CSV中的全部字段名称。
target_column = 'purchase_segment'  # 指定要预测的三分类标签列。
split_column = 'split'  # 指定训练、验证和推理分区列。
id_columns = ['customer_id']  # 定义仅用于对齐结果的唯一业务ID。
private_text_columns = ['full_name', 'street_address', 'phone']  # 定义不应直接建模的高基数隐私文本字段。
date_columns = ['purchase_time']  # 定义需要从字符串解析的日期时间字段。
numeric_columns = ['purchase_quantity', 'unit_price', 'returns_last_year']  # 定义原始数值字段。
categorical_columns = ['city', 'postal_code', 'country', 'country_calling_code', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 定义业务类别字段。
raw_feature_names = [column for column in all_columns if column not in [target_column, split_column]]  # 列出标签和分区以外的所有原始候选特征。
declared_columns = id_columns + private_text_columns + date_columns + numeric_columns + categorical_columns + [target_column, split_column]  # 汇总已经人工确定语义的全部字段。
undeclared_columns = sorted(set(all_columns) - set(declared_columns))  # 找出遗漏业务角色定义的字段。
original_dtypes = frame.dtypes.astype(str)  # 保存CSV刚读入时Pandas推断的原始dtype。
parsed_dates = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 安全解析购买时间并把无效字符串转为NaT。
invalid_date_count = int((frame['purchase_time'].notna() & parsed_dates.isna()).sum())  # 统计原本非空但无法解析的日期数量。
semantic_role = {column: 'numeric' for column in numeric_columns}  # 为数值字段建立业务类型映射。
semantic_role.update({column: 'categorical' for column in categorical_columns})  # 把业务类别字段加入类型映射。
semantic_role.update({column: 'datetime' for column in date_columns})  # 把日期字段加入类型映射。
semantic_role.update({column: 'identifier' for column in id_columns})  # 把唯一ID加入类型映射。
semantic_role.update({column: 'private_high_cardinality_text' for column in private_text_columns})  # 把隐私文本加入类型映射。
semantic_role.update({target_column: 'target', split_column: 'data_partition'})  # 标记标签和数据分区字段。
schema_audit = pd.DataFrame({'column': all_columns})  # 创建一行对应一个字段的结构审计表。
schema_audit['pandas_dtype'] = schema_audit['column'].map(original_dtypes)  # 添加Pandas实际读取dtype。
schema_audit['semantic_role'] = schema_audit['column'].map(semantic_role)  # 添加人工判断的业务语义类型。
schema_audit['non_null_count'] = schema_audit['column'].map(frame.notna().sum())  # 统计每列非空数量。
schema_audit['missing_count'] = schema_audit['column'].map(frame.isna().sum())  # 统计每列NA数量。
schema_audit['missing_rate'] = schema_audit['missing_count'] / len(frame)  # 计算每列NA比例。
schema_audit['unique_count'] = schema_audit['column'].map(frame.nunique(dropna=True))  # 统计每列非NA唯一值数量。
schema_audit['unique_rate'] = schema_audit['unique_count'] / schema_audit['non_null_count'].clip(lower=1)  # 计算非空值中的唯一值比例。
schema_audit['model_action'] = schema_audit['semantic_role'].map({'numeric': 'fill NA then numeric feature', 'categorical': 'fill NA then one-hot encode', 'datetime': 'parse and extract time features', 'identifier': 'exclude; retain only for result join', 'private_high_cardinality_text': 'exclude for privacy and overfitting risk', 'target': 'prediction target; never use as feature', 'data_partition': 'split only; never use as feature'})  # 给每类字段添加建模处理建议。
missing_summary = schema_audit.loc[schema_audit['missing_count'].gt(0), ['column', 'semantic_role', 'missing_count', 'missing_rate']].sort_values(['missing_count', 'column'], ascending=[False, True])  # 筛选并排序所有存在NA的字段。
category_levels = {column: frame[column].value_counts(dropna=False).rename_axis('value').reset_index(name='count') for column in categorical_columns + [target_column, split_column]}  # 为每个类别、标签和分区字段统计所有取值频数。
numeric_summary = frame[numeric_columns].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T  # 汇总数值列数量、均值、标准差、分位数和范围。
numeric_summary['missing_count'] = frame[numeric_columns].isna().sum()  # 在数值统计表中加入NA数量。
numeric_summary['zero_count'] = frame[numeric_columns].eq(0).sum()  # 统计原始数值中真实0的数量以区别于填补值。
numeric_correlation = frame[numeric_columns].corr(numeric_only=True)  # 计算数值特征间Pearson相关系数。
outlier_counts = {}  # 准备保存每个数值字段的IQR异常值数量。
for column in numeric_columns:  # 逐个检查数值字段的异常值。
    values = frame[column].dropna()  # 暂时排除NA后计算有效分位数。
    first_quartile, third_quartile = values.quantile([0.25, 0.75])  # 取得第一和第三四分位数。
    interquartile_range = third_quartile - first_quartile  # 计算四分位距IQR。
    lower_bound = first_quartile - 1.5 * interquartile_range  # 计算常用IQR异常下界。
    upper_bound = third_quartile + 1.5 * interquartile_range  # 计算常用IQR异常上界。
    outlier_counts[column] = int(((values < lower_bound) | (values > upper_bound)).sum())  # 统计超出上下界的数值数量。
outlier_summary = pd.Series(outlier_counts, name='iqr_outlier_count').to_frame()  # 把异常值字典转换成可读表格。
date_summary = pd.DataFrame({'column': date_columns, 'valid_count': [int(parsed_dates.notna().sum())], 'missing_or_invalid_count': [int(parsed_dates.isna().sum())], 'invalid_non_empty_count': [invalid_date_count], 'minimum': [parsed_dates.min()], 'maximum': [parsed_dates.max()]})  # 汇总日期有效量、缺失量、无效量和时间范围。
duplicate_row_count = int(frame.duplicated().sum())  # 统计整行完全重复的记录数量。
duplicate_id_count = int(frame['customer_id'].duplicated().sum())  # 统计重复客户ID数量。
constant_columns = [column for column in all_columns if frame[column].nunique(dropna=False) <= 1]  # 找出没有区分能力的常量字段。
high_cardinality_columns = schema_audit.loc[(schema_audit['unique_rate'] >= 0.50) & schema_audit['semantic_role'].isin(['categorical', 'identifier', 'private_high_cardinality_text']), 'column'].tolist()  # 找出唯一值比例至少50%的高基数字段。
split_distribution = frame['split'].value_counts(dropna=False).rename_axis('split').reset_index(name='row_count')  # 统计训练、验证和推理样本数量。
label_by_split = pd.crosstab(frame['split'], frame[target_column], dropna=False, margins=True)  # 交叉统计每个分区中的标签数量。
train_labels = frame.loc[frame['split'].eq('train'), target_column].dropna()  # 取得训练分区有效标签。
train_label_distribution = train_labels.value_counts().rename_axis('class').reset_index(name='count')  # 统计训练集中每个类别数量。
train_label_distribution['rate'] = train_label_distribution['count'] / len(train_labels)  # 计算训练类别比例以检查不平衡。
numeric_by_class = frame.loc[frame[target_column].notna()].groupby(target_column)[numeric_columns].mean().round(2)  # 比较不同标签的平均数值特征。
inference_target_missing = int(frame.loc[frame['split'].eq('inference'), target_column].isna().sum())  # 确认推理分区没有真实标签。
unexpected_target_missing = int(frame.loc[~frame['split'].eq('inference'), target_column].isna().sum())  # 检查训练或验证数据是否意外缺少标签。
leakage_risks = {'purchase_segment': '目标列，必须从X中排除', 'split': '人工分区标记，不能作为预测特征', 'customer_id': '唯一标识符，可能让模型记住样本', 'full_name/street_address/phone': '隐私且高基数，容易过拟合', 'fill_or_scale_before_split': '使用全数据统计量会让验证和推理信息泄漏到训练'}  # 汇总常见的数据泄漏与建模风险。
model_numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 推荐数值特征并包含日期派生结果。
model_categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 推荐经过降基数处理的类别特征。
excluded_features = {'customer_id': 'only join predictions back to customers', 'full_name': 'private and nearly unique', 'street_address': 'private and high cardinality', 'phone': 'private and nearly unique', 'city': 'small data has too many city levels', 'postal_code': 'use lower-cardinality postal_prefix instead', 'country_calling_code': 'derive and clean phone_country_code consistently', 'purchase_time': 'use extracted month/hour/weekend instead', 'purchase_segment': 'target leakage', 'split': 'partition metadata'}  # 说明每个排除字段的原因。
model_recommendations = pd.DataFrame({'model': ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting', 'KNN', 'GaussianNB'], 'when_useful': ['strong interpretable baseline after scaling and one-hot', 'nonlinear interactions and feature importance without numeric scaling', 'small or medium data with scaled features and flexible boundary', 'strong nonlinear tabular baseline but tune carefully', 'simple distance baseline after scaling; weak in high dimensions', 'very fast baseline with strong independence assumptions']})  # 根据当前小型混合表整理候选模型及适用场景。
assert not undeclared_columns and len(schema_audit) == len(all_columns)  # 验证每个CSV字段都有明确业务角色。
assert duplicate_row_count == 0 and duplicate_id_count == 0  # 验证当前样本和客户ID没有重复。
assert unexpected_target_missing == 0 and inference_target_missing == len(frame.query("split == 'inference'"))  # 验证只有推理分区缺少目标标签。
assert set(train_label_distribution['class']) == {'high_value', 'regular', 'occasional'}  # 验证训练集包含全部三个目标类别。
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):  # 临时允许完整显示所有审计结果。
    print('all_columns:', all_columns, 'raw_feature_names:', raw_feature_names, 'schema_audit:', schema_audit, 'missing_summary:', missing_summary, sep='\n')  # 输出字段列表、角色、dtype、NA和唯一值审计。
    for column, counts in category_levels.items():  # 逐个遍历所有类别字段的类别频数表。
        print(f'category_levels[{column}]:', counts, sep='\n')  # 输出当前字段有哪些类别以及各类数量。
    print('numeric_summary:', numeric_summary, 'numeric_correlation:', numeric_correlation, 'outlier_summary:', outlier_summary, 'date_summary:', date_summary, sep='\n')  # 输出数值、相关性、异常值和日期统计。
    print('duplicate_rows:', duplicate_row_count, 'duplicate_customer_ids:', duplicate_id_count, 'constant_columns:', constant_columns, 'high_cardinality_columns:', high_cardinality_columns, sep='\n')  # 输出重复、常量和高基数字段检查。
    print('split_distribution:', split_distribution, 'label_by_split:', label_by_split, 'train_label_distribution:', train_label_distribution, 'numeric_by_class:', numeric_by_class, sep='\n')  # 输出分区、标签平衡和按类数值差异。
    print('leakage_risks:', leakage_risks, 'recommended_numeric_features:', model_numeric_features, 'recommended_categorical_features:', model_categorical_features, 'excluded_features:', excluded_features, 'model_recommendations:', model_recommendations, sep='\n')  # 输出最终特征方案、排除理由和候选模型建议。
# 自动dtype只反映CSV解析结果，邮编、区号等即使只含数字也仍应根据业务语义作为类别处理。
# 真实项目还应结合业务成本、样本采集方式、公平性要求和时间漂移决定最终模型及验证策略。
