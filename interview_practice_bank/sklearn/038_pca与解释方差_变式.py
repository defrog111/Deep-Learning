"""
题目 038：PCA与解释方差_变式

要求：完成“PCA与解释方差”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. PCA前标准化不同量纲特征。
2. 加载数据。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 大数据用增量PCA；whiten配置会把主成分缩放到单位方差。

完成标准：
- 验证增量投影和白化配置。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入高维数据。
from sklearn.decomposition import PCA  # 导入PCA。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # PCA前标准化不同量纲特征。
features, _ = load_breast_cancer(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.decomposition import IncrementalPCA  # 导入可分批学习的PCA。
stable_pca_features = features.astype(np.float32); incremental = IncrementalPCA(n_components=2, batch_size=50); incremental.fit(stable_pca_features); incremental_projection = incremental.transform(stable_pca_features[:5]); whitening_configuration = PCA(n_components=2, whiten=True, svd_solver='randomized', random_state=42)  # 大数据用增量PCA；whiten配置会把主成分缩放到单位方差。
assert incremental_projection.shape == (5, 2) and whitening_configuration.whiten and whitening_configuration.svd_solver == 'randomized'  # 验证增量投影和白化配置。
