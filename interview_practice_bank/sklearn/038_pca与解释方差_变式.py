"""
题目 038：PCA与解释方差_变式

要求：完成“PCA与解释方差”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. PCA前标准化不同量纲特征。
2. 加载数据。
3. 标准化后使用完整SVD降维。
4. 拟合主成分并投影。
5. 取得已拟合PCA步骤。

完成标准：
- 验证降维shape和解释方差。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入高维数据。
from sklearn.decomposition import PCA  # 导入PCA。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # PCA前标准化不同量纲特征。
features, _ = load_breast_cancer(return_X_y=True)  # 加载数据。
pipeline = make_pipeline(StandardScaler(), PCA(n_components=4, svd_solver='full'))  # 标准化后使用完整SVD降维。
projected = pipeline.fit_transform(features)  # 拟合主成分并投影。
pca = pipeline.named_steps['pca']  # 取得已拟合PCA步骤。
assert projected.shape[1] == 4 and pca.explained_variance_ratio_.sum() <= 1  # 验证降维shape和解释方差。
print(projected.shape, pca.explained_variance_ratio_)  # 输出PCA结果。
