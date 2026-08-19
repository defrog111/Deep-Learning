# Python / Data / ML 高频题库

本目录包含 668 道独立、可运行的练习，其中原题库550题、CSV专项76题、矩阵运算
强化42题。每一个题目
`.py` 文件只放一道题，文件顶部包含
中文要求、逐步操作过程和完成标准，后面是带逐行中文注释的参考代码。

## 题量

| 目录 | 数量 | 重点 |
|---|---:|---|
| `pandas/` | 100 | CSV、索引、清洗、GroupBy、连接、重塑、时间序列和易错点 |
| `numpy/` | 100 | ndarray、广播、axis、矩阵运算、线性代数、FFT和内存 |
| `pytorch/` | 100 | Tensor、Autograd、训练、模型、CNN/RNN/Attention/Transformer |
| `python/` | 100 | 数据结构、函数、OOP、迭代器、装饰器、标准库和复杂度 |
| `ml/` | 100 | 损失、正则化、验证、指标、经典算法、漂移和可复现性 |
| `sklearn/` | 50 | Pipeline、预处理、CV、调参、模型、评估和持久化 |
| `csv_data_processing/` | 76 | 每题从CSV读取，覆盖清洗、连接、时间序列、特征工程和模型数据管道 |
| `matrix_operations/numpy/` | 21 | NumPy 矩阵乘法、方程、分解、数值稳定性和 ML 应用 |
| `matrix_operations/pytorch/` | 21 | PyTorch 批量矩阵、Autograd、Attention、稀疏矩阵和线性代数 |

“包含所有知识点”无法做数学意义上的穷尽；这里覆盖的是面试、笔试和日常 ML 开发中
最常考、最容易写错的主干知识。Python、NumPy、PyTorch 和 ML 的每个主知识点提供
基础、变式、易错点和综合四种不同的可执行版本；scikit-learn 每个主知识点提供基础和
综合两种不同版本。同一核心API会以位置参数、关键字参数、不同shape和不同业务场景反复出现。

## 安装

```bash
cd interview_practice_bank
python -m pip install -r requirements.txt
```

## 使用方法

建议先只看文件顶部题目，遮住下面参考代码，自己在新文件里实现。写完后运行原题核对：

```bash
python pandas/005_读取csv与基本检查_基础.py
python numpy/057_线性方程组求解_基础.py
python pytorch/085_multiheadattention_基础.py
```

文件名由“编号 + 知识点 + 难度”组成。推荐顺序：

1. Python
2. NumPy
3. Pandas
4. ML 原理
5. scikit-learn
6. PyTorch

## 示例数据

- `data/sales.csv`：销售、日期、类别、数量、价格、折扣。
- `data/employees.csv`：员工、部门、城市、工资和经理关系。
- `data/house_prices.csv`：用于NumPy、PyTorch和scikit-learn线性回归的房屋特征与价格。
- `data/classification_examples.csv`：带train、val、inference固定分区的二分类、多分类和多标签数据。
- `data/customer_churn_interview.csv`：面试用混合类型、缺失值、日期、业务ID和固定分区的流失数据。
- `data/time_series_sequences.csv`：十二步历史窗口、趋势类别、下一时刻目标和固定数据分区。
- `data/multimodal_products.csv`：商品描述文本、价格评分库存数值、类别标签和固定数据分区。
- `data/customer_purchases_mixed.csv`：姓名、地址、邮编、国家区号、电话、购买时间与数量等真实风格混合字段分类数据。

CSV 题目通过 `Path(__file__)` 定位数据，因此从仓库根目录或题库目录运行都可以。

CSV专项还提供客户、脏订单、传感器和评论数据；详见
[`csv_data_processing/README.md`](csv_data_processing/README.md)。

## 自动检查

检查数量、Python 语法以及每行参考代码是否带注释：

```bash
python validate_bank.py
```

检查 Python、NumPy、PyTorch、ML 和 scikit-learn 的必备API与高频考点覆盖：

```bash
python validate_coverage.py
```

检查同一知识点的不同难度是否只是修改数字、字符串、`print` 或 `assert`：

```bash
python audit_variant_quality.py
```

覆盖清单目前包含 359 项强制检查：Python 57 项、NumPy 74 项、PyTorch 85 项、
ML 82 项、scikit-learn 61 项。重复度审计针对本次全面重写的这五类目录；Pandas
仍由结构、语法、注释和运行检查负责。

执行某一类或全部题目：

```bash
python run_all.py --category pandas
python run_all.py --category all
```

检查并运行69个CSV专项例子：

```bash
python validate_csv_examples.py
```

检查并运行42个 NumPy/PyTorch 矩阵运算强化题：

```bash
python validate_matrix_drills.py
python run_matrix_drills.py --category all
```

需要把任何 warning 也当作失败时，可以运行：

```bash
PYTHONWARNINGS=error python run_all.py --category all
```

默认隐藏每题输出，只显示进度。需要观察全部结果时增加 `--show-output`。

## 重新生成

`create_bank.py` 和各个 `*_cases.py` 保存了题库定义。重新生成会删除六个题目目录并重建，
但不会删除 `data/`、README 或其他项目代码：

```bash
python create_bank.py
```

矩阵强化题由 `matrix_cases.py` 单独保存，重新生成不会影响原来的六类题库：

```bash
python create_matrix_drills.py
```
