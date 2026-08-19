"""
CSV数据处理练习 070：混合客户表Logistic分类

题目：读取包含姓名、地址、邮编、电话、购买时间、数量和类别字段的客户CSV，完成特征工程后用Logistic Regression预测购买分层。

操作过程：
1. 读取并清理真实风格的混合字段客户表。
2. 从时间、邮编和电话提取可泛化特征。
3. 排除姓名、完整地址、完整电话和客户ID。
4. 分别填补、标准化和独热编码数值与类别特征。
5. 只用训练分区拟合模型并评估验证分区。
6. 为无标签推理分区输出类别和概率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取和处理混合字段CSV。
from sklearn.compose import ColumnTransformer  # 导入按列类型分别预处理的工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入多分类Logistic Regression。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入常用分类指标。
from sklearn.pipeline import Pipeline  # 导入防止预处理数据泄漏的流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入类别独热编码和数值标准化工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_purchases_mixed.csv'  # 定位包含多种业务字段的客户购买CSV。
frame = pd.read_csv(csv_path)  # 从磁盘读取完整表格。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全空记录并重建索引。
frame['purchase_time'] = pd.to_datetime(frame['purchase_time'], errors='coerce')  # 把购买时间安全解析为日期时间。
frame['purchase_month'] = frame['purchase_time'].dt.month  # 提取月份作为季节性数值特征。
frame['purchase_hour'] = frame['purchase_time'].dt.hour  # 提取小时作为购买习惯特征。
frame['is_weekend'] = frame['purchase_time'].dt.dayofweek.ge(5).astype(int)  # 把周末购买编码为0或1。
frame['postal_prefix'] = frame['postal_code'].astype('string').str[:2]  # 只保留邮编前缀以降低类别基数。
frame['phone_country_code'] = frame['phone'].str.extract(r'^\+(\d+)-', expand=False)  # 从电话中提取国家区号而不使用完整号码。
numeric_na_columns = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour']  # 指定允许用0表示缺失的数值和时间派生列。
categorical_na_columns = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 指定需要未知类别标记的字符串列。
frame[numeric_na_columns] = frame[numeric_na_columns].fillna(0)  # 示例策略：把数值NA统一填成0以保证模型能够读取。
frame[categorical_na_columns] = frame[categorical_na_columns].fillna('unknown')  # 把类别NA填为unknown而不是错误地填成数字0。
train_frame = frame.query("split == 'train'").copy()  # 取得有标签训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得有标签验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得无标签推理分区。
numeric_features = ['purchase_quantity', 'unit_price', 'returns_last_year', 'purchase_month', 'purchase_hour', 'is_weekend']  # 定义数值模型特征。
categorical_features = ['country', 'phone_country_code', 'postal_prefix', 'product_category', 'payment_method', 'membership_level', 'device_type']  # 定义低到中基数类别特征。
features = numeric_features + categorical_features  # 汇总模型输入并主动排除身份及隐私字段。
# drop变体：feature_frame = frame.drop(columns=['customer_id', 'full_name', 'street_address', 'city', 'postal_code', 'country_calling_code', 'phone', 'purchase_time', 'purchase_segment', 'split'])  # 按列名排除不直接建模的字段。
# iloc变体：raw_feature_frame = frame.iloc[:, 5:16]  # 按位置选择country到returns_last_year，但仍需继续处理电话和时间等字段。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])  # 用训练中位数填补数值并标准化。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # 用训练众数填补类别并独热编码。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 组合两类预处理分支。
model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])  # 串联预处理和多分类线性模型。
model.fit(train_frame[features], train_frame['purchase_segment'])  # 仅用训练分区拟合所有预处理参数和分类器。
val_predictions = model.predict(val_frame[features])  # 预测验证集的唯一购买分层。
val_accuracy = accuracy_score(val_frame['purchase_segment'], val_predictions)  # 计算验证准确率。
val_precision = precision_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 等权计算各类precision。
val_recall = recall_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 等权计算各类recall。
val_f1 = f1_score(val_frame['purchase_segment'], val_predictions, average='macro', zero_division=0)  # 等权计算各类F1。
inference_predictions = model.predict(inference_frame[features])  # 为无标签客户预测类别。
inference_probabilities = model.predict_proba(inference_frame[features]).max(axis=1)  # 取得每行预测类别对应的最大概率。
inference_result = inference_frame[['customer_id', 'full_name']].assign(predicted_segment=inference_predictions, confidence=inference_probabilities)  # 把预测与业务标识重新对齐。
assert len(inference_result) == len(inference_frame) and inference_result['confidence'].between(0, 1).all()  # 验证推理数量和概率范围。
print('accuracy:', val_accuracy, 'precision_macro:', val_precision, 'recall_macro:', val_recall, 'f1_macro:', val_f1, 'inference:', inference_result, sep='\n')  # 输出验证指标和推理结果。
# 姓名、完整地址、电话和客户ID通常是高基数身份字段，直接独热编码容易记住个人且难以泛化。
