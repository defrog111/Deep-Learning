"""
CSV数据处理练习 074：混合客户表KNN分类

题目：读取含多种列类型的客户购买CSV，正确编码和缩放特征后用KNN按邻居完成购买分层分类。

操作过程：
1. 从日期、邮编和电话中创建模型特征。
2. 排除不适合直接计算距离的身份字段。
3. 填补、标准化数值列并独热编码类别列。
4. 用五个最近邻预测类别并计算验证指标。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas。
from sklearn.compose import ColumnTransformer  # 导入分列转换器。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入分类指标。
from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入独热编码器和标准化器。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取所有字段。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除全空记录并整理索引。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 安全解析购买时间。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 创建月份数值列。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 创建小时数值列。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 创建周末标记。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 使用较粗粒度邮编前缀。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从电话号码提取国家区号。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定可以用0填补的数值列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定需要unknown标记的类别列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 把数值和时间派生NA显式填成0。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 把类别NA显式填成unknown。
train_frame = frame.query("split == 'train'").copy()  # 取得训练数据。
val_frame = frame.query("split == 'val'").copy()  # 取得验证数据。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理数据。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 指定数值列。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定类别列。
features = numeric_features + categorical_features  # 汇总输入列并排除高基数个人字段。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 按名称排除非模型字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 按位置选择原始候选字段。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])  # 填补并统一数值特征尺度。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # 填补并编码类别特征。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 合并数值和类别转换。
model = Pipeline([('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))])  # 创建按距离加权的五邻居分类器。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 只用训练分区保存预处理参数和训练样本。
val_predictions = model.predict(val_frame[features])  # 根据最近训练邻居预测验证类别。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro F1。
inference_predictions = model.predict(inference_frame[features])  # 预测推理类别。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions)  # 生成带客户标识的推理结果。
assert len(inference_result) == len(inference_frame) and 1 <= model.named_steps['classifier'].n_neighbors <= len(train_frame)  # 验证输出数量和邻居数合法。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'inference:', inference_result, sep='\n')  # 输出指标和预测。
# KNN直接比较距离，所以数值列若不标准化，unit_price等大尺度特征会不合理地主导距离。
