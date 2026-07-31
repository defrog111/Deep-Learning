"""
CSV数据处理练习 037：DataFrame转NumPy矩阵

题目：读取Iris CSV，把数值特征转换为NumPy矩阵，完成标准化和协方差计算。

操作过程：
1. 定位项目中的Iris CSV。
2. 从CSV读取Iris。
3. 选择数值列并复制为float32矩阵。
4. 按特征标准化。
5. 计算特征协方差矩阵。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import numpy as np  # 导入NumPy。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[2] / 'transformer_learning' / 'data' / 'iris.csv'  # 定位项目中的Iris CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取Iris。
matrix = frame.select_dtypes(include='number').to_numpy(dtype=np.float32, copy=True)  # 选择数值列并复制为float32矩阵。
standardized = (matrix - matrix.mean(axis=0, keepdims=True)) / matrix.std(axis=0, keepdims=True)  # 按特征标准化。
covariance = np.cov(standardized, rowvar=False)  # 计算特征协方差矩阵。
assert matrix.shape[1] == 4 and np.allclose(standardized.mean(0), 0, atol=1e-6) and covariance.shape == (4, 4)  # 验证矩阵和统计结果。
print(matrix.shape, covariance, sep='\n')  # 输出shape和协方差。
