"""
CSV数据处理练习 071：混合客户表RandomForest分类

题目：读取真实风格混合客户表，处理时间、数值和类别列，并用Random Forest预测三种购买分层。

操作过程：
1. 读取混合字段CSV并构造时间和字符串派生特征。
2. 排除高基数身份字段。
3. 用训练统计量填补缺失值并编码类别。
4. 训练随机森林并计算多分类指标。
5. 输出推理类别及特征重要性。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
from sklearn.compose import ColumnTransformer  # 导入分列预处理工具。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入多分类指标。
from sklearn.pipeline import Pipeline  # 导入完整模型流水线。
from sklearn.preprocessing import OneHotEncoder  # 导入类别独热编码器。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位混合客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取完整表格。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全空记录。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 安全解析购买时间。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 提取购买月份。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 提取购买小时。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 创建周末标记。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 从邮编提取较低基数的区域前缀。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从完整电话提取国家区号。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定使用0填补的数值和时间派生列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定使用unknown填补的类别列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 显式把数值NA填成0。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 显式把类别NA填成unknown类别。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 定义数值特征。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 定义类别特征。
features = numeric_features + categorical_features  # 汇总模型输入列并排除姓名、地址、完整电话和ID。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 按名称排除非模型字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 按位置选择一段原始候选特征，之后仍需特征工程。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])  # 树模型只填补数值缺失而不要求标准化。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # 填补并生成稠密独热类别特征。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 组合数值与类别处理。
model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))])  # 创建包含100棵树的可复现随机森林。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 只用训练数据拟合流水线。
val_predictions = model.predict(val_frame[features])  # 预测验证类别。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro F1。
inference_predictions = model.predict(inference_frame[features])  # 预测推理客户的购买分层。
feature_names = model.named_steps['preprocessor'].get_feature_names_out()  # 取得预处理后的全部特征名。
importance = pd.Series(model.named_steps['classifier'].feature_importances_, index=feature_names).nlargest(8)  # 找出最重要的八个特征。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions)  # 对齐客户和预测结果。
assert len(inference_result) == len(inference_frame) and len(importance) == 8  # 验证推理数量与重要性输出。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'top_importance:', importance, 'inference:', inference_result, sep='\n')  # 输出指标、重要性和推理结果。
# Random Forest按特征阈值切分，因此数值特征通常不需要StandardScaler。
