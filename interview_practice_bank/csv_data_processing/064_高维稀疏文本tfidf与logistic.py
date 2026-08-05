"""
CSV数据处理练习 064：高维稀疏文本TFIDF与Logistic

题目：分别从CSV读取train、val和inference商品描述，用一元和二元TF-IDF构造高维稀疏矩阵，再用Logistic Regression进行多分类概率预测并查看类别关键词。

操作过程：
1. 定位含description文本的CSV。
2. 从CSV读取训练文本和标签。
3. 从CSV读取验证文本和标签。
4. 从CSV读取推理文本。
5. 串联高维稀疏TF-IDF和多分类Logistic。
6. 只用训练描述学习词表、IDF和分类参数。
7. 取得训练文本对应的CSR稀疏矩阵。
8. 输出验证文本的多类别概率。
9. 将最大概率列映射为类别名称。
10. 计算验证准确率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取文本CSV。
from scipy import sparse  # 导入稀疏矩阵检查工具。
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF文本特征提取器。
from sklearn.linear_model import LogisticRegression  # 导入支持稀疏输入的线性分类器。
from sklearn.metrics import accuracy_score, log_loss  # 导入准确率和概率质量指标。
from sklearn.pipeline import Pipeline  # 导入文本流水线。
csv_path = Path(__file__).parents[1] / 'data' / 'multimodal_products.csv'  # 定位含description文本的CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练文本和标签。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证文本和标签。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理文本。
model = Pipeline([('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True)), ('classifier', LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced'))])  # 串联高维稀疏TF-IDF和多分类Logistic。
model.fit(train_frame['description'], train_frame['category'])  # 只用训练描述学习词表、IDF和分类参数。
train_matrix = model.named_steps['tfidf'].transform(train_frame['description'])  # 取得训练文本对应的CSR稀疏矩阵。
val_probabilities = model.predict_proba(val_frame['description'])  # 输出验证文本的多类别概率。
val_predictions = model.classes_[val_probabilities.argmax(axis=1)]  # 将最大概率列映射为类别名称。
val_accuracy = accuracy_score(val_frame['category'], val_predictions)  # 计算验证准确率。
val_log_loss = log_loss(val_frame['category'], val_probabilities, labels=model.classes_)  # 评估预测概率是否把足够质量放在真实类别。
feature_names = model.named_steps['tfidf'].get_feature_names_out()  # 取得一元和二元文本特征名称。
classifier = model.named_steps['classifier']  # 取得已拟合Logistic分类器。
top_terms = {class_name: feature_names[classifier.coef_[index].argsort()[-5:][::-1]].tolist() for index, class_name in enumerate(classifier.classes_)}  # 提取每个类别正系数最大的五个关键词。
inference_probabilities = model.predict_proba(inference_frame['description'])  # 对CSV推理文本输出类别概率。
inference_labels = model.classes_[inference_probabilities.argmax(axis=1)]  # 生成推理类别。
inference_result = inference_frame[['item_id', 'description']].assign(predicted_category=inference_labels)  # 将文本预测与商品ID对齐。
assert sparse.issparse(train_matrix) and train_matrix.shape[1] == len(feature_names) and val_accuracy >= 0.80  # 验证全程保持稀疏表示且模型有效。
print('sparse_shape:', train_matrix.shape, 'stored_nonzero_values:', train_matrix.nnz, 'validation_accuracy:', val_accuracy, 'validation_log_loss:', val_log_loss, 'top_terms:', top_terms, 'inference:', inference_result, sep='\n')  # 输出稀疏结构、指标、关键词和推理结果。
# 高频考点：TF-IDF矩阵通常极高维但大部分为0，不能随意调用toarray转换为密集矩阵。
# Logistic Regression提供predict_proba，适合需要阈值、概率排序或概率解释的任务。
# ngram_range=(1,2)同时使用单词和相邻双词；更大的ngram会增加维度、内存和过拟合风险。
