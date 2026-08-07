"""
CSV数据处理练习 068：SVM读取CSV分类

题目：读取数值分类CSV，使用训练分区拟合最简单的线性SVM，并评估验证准确率及生成推理类别。

操作过程：
1. 定位数值分类CSV。
2. 读取并清理整行全空记录。
3. 取得训练、验证和推理分区。
4. 选择四个数值特征和多分类标签。
5. 只用训练数据拟合线性SVM。
6. 计算验证准确率并预测推理类别。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.metrics import accuracy_score  # 导入分类准确率指标。
from sklearn.svm import SVC  # 导入支持向量分类器。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位数值分类CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部样本。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建索引。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定四个数值分类特征。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
model = SVC(kernel='linear')  # 创建最简单的线性支持向量分类器。
model.fit(train_frame[feature_names], train_frame['class_label'])  # 只用训练特征和标签拟合分类边界。
val_predictions = model.predict(val_frame[feature_names])  # 预测验证类别。
val_accuracy = accuracy_score(val_frame['class_label'], val_predictions)  # 计算验证准确率。
inference_predictions = model.predict(inference_frame[feature_names])  # 预测推理类别。
assert val_accuracy >= 0.80 and len(inference_predictions) == len(inference_frame)  # 验证分类效果和推理数量。
print('validation_accuracy:', val_accuracy, 'inference_predictions:', inference_predictions, sep='\n')  # 输出验证指标和推理类别。
# SVM对特征尺度敏感，真实量纲差异较大时可在模型前增加StandardScaler。
