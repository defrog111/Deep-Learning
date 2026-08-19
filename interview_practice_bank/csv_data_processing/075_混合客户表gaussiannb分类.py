"""
CSV数据处理练习 075：混合客户表GaussianNB分类

题目：读取真实风格混合客户表，把不同字段转换成稠密数值矩阵，并用Gaussian Naive Bayes预测购买分层。

操作过程：
1. 清洗CSV并提取日期、邮编及国家区号特征。
2. 排除姓名、地址、客户ID和完整电话。
3. 填补与编码混合字段并输出稠密矩阵。
4. 训练GaussianNB并输出验证指标和推理概率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
from sklearn.compose import ColumnTransformer  # 导入按字段类型转换工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入分类指标。
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯分类器。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入类别编码和数值标准化工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位混合客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取表格。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全空数据。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 安全解析购买时间。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 提取月份。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 提取小时。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 创建周末标记。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 从邮编提取区域前缀。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从完整电话提取国家区号。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定用0表示缺失的数值列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定用unknown表示缺失的类别列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 把数值和日期派生NA显式填成0。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 把类别NA显式填成unknown。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 定义数值特征。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 定义类别特征。
features = numeric_features + categorical_features  # 汇总建模字段并排除身份字段。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 按名称排除字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 按位置选取原始候选字段。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])  # 用训练中位数填补并标准化数值。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # 填补类别并输出稠密独热矩阵。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 合并全部数值化特征。
model = Pipeline([('preprocessor', preprocessor), ('classifier', GaussianNB(var_smoothing=1e-8))])  # 创建带方差平滑的高斯朴素贝叶斯流水线。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 只用训练数据估计各类别特征分布。
val_predictions = model.predict(val_frame[features])  # 预测验证类别。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro F1。
inference_predictions = model.predict(inference_frame[features])  # 预测推理客户类别。
inference_probabilities = model.predict_proba(inference_frame[features]).max(axis=1)  # 取得预测类别最大后验概率。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions, confidence=inference_probabilities)  # 对齐客户与推理输出。
assert len(inference_result) == len(inference_frame) and inference_result['confidence'].between(0, 1).all()  # 验证输出完整且概率有效。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'inference:', inference_result, sep='\n')  # 输出验证指标和推理结果。
# GaussianNB假设给定类别后各特征条件独立且近似高斯，假设较强但训练和预测都很快。
