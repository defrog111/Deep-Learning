"""
CSV数据处理练习 065：高维稀疏文本TFIDF与LinearSVM

题目：分别从CSV读取train、val和inference商品描述，用TF-IDF稀疏矩阵和LinearSVC完成多分类，并解释margin分数、C参数以及为什么没有原生predict_proba。

操作过程：
1. 定位文本分类CSV。
2. 从CSV读取训练文本。
3. 从CSV读取验证文本。
4. 从CSV读取推理文本。
5. 串联稀疏TF-IDF和最大间隔线性SVM。
6. 只在训练分区学习词表、IDF和最大间隔超平面。
7. 检查传入LinearSVC的是CSR稀疏特征。
8. 直接输出验证类别。
9. 输出到各类别超平面的有符号margin分数。
10. 计算验证准确率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from scipy import sparse  # 导入稀疏矩阵检查工具。
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入文本向量化工具。
from sklearn.metrics import accuracy_score, classification_report  # 导入分类评估指标。
from sklearn.pipeline import Pipeline  # 导入文本建模流水线。
from sklearn.svm import LinearSVC  # 导入适合高维稀疏文本的线性支持向量机。
csv_path = Path(__file__).parents[1] / 'data' / 'multimodal_products.csv'  # 定位文本分类CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练文本。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证文本。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理文本。
model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True)), ('classifier', LinearSVC(C=1.0, class_weight='balanced', random_state=42))])  # 串联稀疏TF-IDF和最大间隔线性SVM。
model.fit(train_frame['description'], train_frame['category'])  # 只在训练分区学习词表、IDF和最大间隔超平面。
train_matrix = model.named_steps['tfidf'].transform(train_frame['description'])  # 检查传入LinearSVC的是CSR稀疏特征。
val_predictions = model.predict(val_frame['description'])  # 直接输出验证类别。
val_decision_scores = model.decision_function(val_frame['description'])  # 输出到各类别超平面的有符号margin分数。
val_accuracy = accuracy_score(val_frame['category'], val_predictions)  # 计算验证准确率。
report = classification_report(val_frame['category'], val_predictions, output_dict=True, zero_division=0)  # 计算各类别precision、recall和F1。
inference_labels = model.predict(inference_frame['description'])  # 对推理文本生成最终类别。
inference_scores = model.decision_function(inference_frame['description'])  # 输出推理margin而不是概率。
inference_result = inference_frame[['item_id', 'description']].assign(predicted_category=inference_labels, winning_margin=inference_scores.max(axis=1))  # 将预测类别和最大margin与商品ID对齐。
assert sparse.issparse(train_matrix) and val_decision_scores.shape == (len(val_frame), len(model.classes_)) and val_accuracy >= 0.80  # 验证稀疏输入、margin shape和效果。
print('sparse_shape:', train_matrix.shape, 'validation_accuracy:', val_accuracy, 'classification_report:', report, 'inference:', inference_result, sep='\n')  # 输出稀疏结构、验证指标和推理结果。
# LinearSVC优化最大间隔且非常适合高维稀疏文本，很多文本任务中会比非线性核SVM更实用。
# decision_function是margin分数而不是概率；需要概率时可用CalibratedClassifierCV在交叉验证中校准。
# C越大越重视训练误差、正则越弱；C越小间隔更平滑、正则更强，必须只用训练交叉验证选择。
