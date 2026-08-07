"""
CSV数据处理练习 067：KMeans读取CSV聚类

题目：读取数值分类CSV，只使用训练分区拟合三个KMeans簇，并为验证和推理数据生成簇编号。

操作过程：
1. 定位数值分类CSV。
2. 读取并清理整行全空记录。
3. 取得训练、验证和推理分区。
4. 选择四个数值特征。
5. 只用训练特征拟合KMeans。
6. 为验证和推理样本预测簇编号。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import os  # 导入环境变量工具以限制本地并行核数探测。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.cluster import KMeans  # 导入KMeans无监督聚类模型。
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')  # 避免受限环境无法探测物理CPU数量时产生警告。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位数值分类CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部样本。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建索引。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定四个数值聚类特征。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
model = KMeans(n_clusters=3, n_init=10, random_state=42)  # 创建可复现的三簇KMeans模型。
train_clusters = model.fit_predict(train_frame[feature_names])  # 只用训练特征学习簇中心并返回训练簇编号。
val_clusters = model.predict(val_frame[feature_names])  # 使用训练簇中心分配验证样本。
inference_clusters = model.predict(inference_frame[feature_names])  # 使用相同簇中心分配推理样本。
assert len(set(train_clusters)) == 3 and len(inference_clusters) == len(inference_frame)  # 验证得到三个簇且推理数量正确。
print('cluster_centers:', model.cluster_centers_, 'validation_clusters:', val_clusters, 'inference_clusters:', inference_clusters, sep='\n')  # 输出簇中心及分区簇编号。
# KMeans是无监督聚类，簇编号0、1、2没有固定类别语义，也不保证对应class_label的原始编号。
