"""
CSV数据处理练习 069：RandomForest读取CSV分类

题目：读取数值分类CSV，使用训练分区拟合简单的二分类随机森林，并评估验证准确率及生成推理类别。

操作过程：
1. 定位数值分类CSV。
2. 读取并清理整行全空记录。
3. 取得训练、验证和推理分区。
4. 选择四个数值特征和二分类标签。
5. 只用训练数据拟合随机森林。
6. 计算验证准确率并预测推理类别。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器。
from sklearn.metrics import accuracy_score  # 导入分类准确率指标。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位数值分类CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部样本。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建索引。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定四个数值分类特征。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)  # 创建限制深度以减少过拟合的可复现随机森林。
model.fit(train_frame[feature_names], train_frame['binary_label'])  # 只用训练特征和二分类标签拟合森林。
val_predictions = model.predict(val_frame[feature_names])  # 预测验证类别。
val_accuracy = accuracy_score(val_frame['binary_label'], val_predictions)  # 计算验证准确率。
inference_predictions = model.predict(inference_frame[feature_names])  # 预测推理类别。
assert val_accuracy >= 0.80 and len(inference_predictions) == len(inference_frame)  # 验证分类效果和推理数量。
print('validation_accuracy:', val_accuracy, 'feature_importances:', model.feature_importances_, 'inference_predictions:', inference_predictions, sep='\n')  # 输出指标、特征重要性和推理类别。
# 随机森林通常不需要StandardScaler，因为树根据单个特征阈值完成切分。
