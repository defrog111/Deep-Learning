"""
CSV数据处理练习 073：混合客户表GradientBoosting分类

题目：读取真实风格客户购买表，对混合字段完成无泄漏预处理，并用Gradient Boosting预测购买分层。

操作过程：
1. 读取CSV并从时间、邮编、电话创建派生特征。
2. 分离数值和类别字段并排除身份字段。
3. 使用稠密独热编码适配Gradient Boosting。
4. 训练模型并输出多分类指标和推理结果。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
from sklearn.compose import ColumnTransformer  # 导入按列预处理工具。
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升分类器。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入分类评估指标。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder  # 导入类别独热编码器。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取混合表格。
frame = frame.dropna(how='all').reset_index(drop=True)  # 清除整行全为空的数据。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 将购买时间转换成datetime。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 提取月份特征。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 提取小时特征。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 标记是否周末购买。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 用邮编前缀表示粗粒度地区。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从完整电话中提取国家区号。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定用0填补的数值和时间列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定用unknown填补的类别列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 示例中把全部数值NA填成0。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 示例中把全部类别NA填成unknown。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 定义数值特征。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 定义类别特征。
features = numeric_features + categorical_features  # 汇总模型列并排除高基数身份字段。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 按名称排除字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 按位置选择原始候选特征区间。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])  # 用训练中位数填补数值缺失。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # 填补类别并输出稠密独热矩阵。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 合并两类预处理结果。
model = Pipeline([('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))])  # 创建逐步修正前序错误的梯度提升模型。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 只用训练分区拟合完整流水线。
val_predictions = model.predict(val_frame[features])  # 预测验证购买分层。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro F1。
inference_predictions = model.predict(inference_frame[features])  # 为无标签分区预测类别。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions)  # 将预测值与客户信息对齐。
assert len(inference_result) == len(inference_frame) and set(inference_predictions).issubset(set(train_frame['purchase_segment']))  # 验证数量和类别范围。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'inference:', inference_result, sep='\n')  # 输出指标与推理结果。
# Gradient Boosting让后续弱树不断拟合前面模型的错误，与随机森林的并行独立建树思路不同。
