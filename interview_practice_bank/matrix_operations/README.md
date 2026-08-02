# NumPy / PyTorch 矩阵运算强化

本目录共 42 道独立练习：NumPy 21 题、PyTorch 21 题。每题都有中文任务、操作
步骤、完成标准、可执行参考代码和逐行注释。

## 覆盖内容

| 模块 | 高频考点 |
|---|---|
| 乘法与 shape | `dot`、`mv`、`mm`、`matmul`、`bmm`、广播、逐元素乘法区别 |
| 张量收缩 | `einsum`、`tensordot`、outer、Kronecker、Attention 的 QKᵀV |
| 方程求解 | `solve`、多右端项、批量方程、`lstsq`、`pinv`、秩亏矩阵 |
| 矩阵分解 | `eig/eigh`、SVD、低秩近似、QR、Cholesky |
| 数值问题 | determinant、`slogdet`、条件数、float32/float64、避免显式求逆 |
| ML 应用 | 标准化、协方差、PCA、线性回归梯度、`nn.Linear`、causal mask |
| PyTorch 专项 | Autograd、稀疏矩阵、融合 `addmm/baddbmm`、device/dtype 思维 |

## 推荐练习顺序

先完成 NumPy 版本，理解数学含义和 shape；再完成 PyTorch 版本，重点关注 batch、
Autograd、dtype/device 和深度学习中的实际用法。

```bash
python matrix_operations/numpy/001_matmul维度规则.py
python matrix_operations/pytorch/017_矩阵Autograd梯度.py
```

批量验证和执行：

```bash
python validate_matrix_drills.py
python run_matrix_drills.py --category numpy
python run_matrix_drills.py --category pytorch
python run_matrix_drills.py --category all
```

需要重新生成题目时运行：

```bash
python create_matrix_drills.py
```
