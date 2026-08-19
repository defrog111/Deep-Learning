"""
CSV数据处理练习 072：混合客户表SVM分类

题目：读取带类别、地址、邮编、电话和时间字段的客户购买CSV，正确预处理后用RBF SVM完成多分类。

操作过程：
1. 从原始字符串和时间列创建可用特征。
2. 排除高基数身份字段并划分固定数据分区。
3. 在Pipeline内部填补、标准化和独热编码。
4. 训练RBF SVM并输出验证指标和推理概率。

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
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入编码与标准化工具。
from sklearn.svm import SVC  # 导入支持向量分类器。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位混合字段CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取所有分区。
frame = frame.dropna(how='all').reset_index(drop=True)  # 清除全空行。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 解析购买时间。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 创建月份特征。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 创建小时特征。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 创建周末二元特征。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 提取邮编前两位。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从电话提取国家区号。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定用0填补的数值和时间派生列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定用unknown填补的类别列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 将数值NA显式替换为0。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 将字符串NA显式替换为unknown。
train_frame = frame.query("split == 'train'").copy()  # 选择训练样本。
val_frame = frame.query("split == 'val'").copy()  # 选择验证样本。
inference_frame = frame.query("split == 'inference'").copy()  # 选择推理样本。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 指定数值列。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定类别列。
features = numeric_features + categorical_features  # 汇总最终特征并排除姓名、地址、电话和客户ID。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 根据列名排除字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 根据位置取得原始候选字段。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])  # 填补并标准化数值特征以适应距离模型。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # 填补并独热编码类别特征。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 创建混合字段预处理器。
model = Pipeline([('preprocessor', preprocessor), ('classifier', SVC(kernel='rbf', C=2.0, probability=True, random_state=42))])  # 创建带概率输出的RBF核SVM。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 仅在训练数据上拟合预处理器和SVM。
val_predictions = model.predict(val_frame[features])  # 预测验证类别。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 计算macro F1。
inference_predictions = model.predict(inference_frame[features])  # 预测推理分区类别。
inference_probabilities = model.predict_proba(inference_frame[features]).max(axis=1)  # 取得每行预测置信度。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions, confidence=inference_probabilities)  # 生成可交付推理表。
assert len(inference_result) == len(inference_frame) and inference_result['confidence'].between(0, 1).all()  # 验证推理结果完整且概率有效。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'inference:', inference_result, sep='\n')  # 输出验证和推理结果。
# SVM依赖样本间距离，购买数量、价格等数值量纲不同，因此StandardScaler很重要。
