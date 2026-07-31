# Python / Data / ML 高频题库

本目录包含 550 道独立、可运行的练习。每一个 `.py` 文件只放一道题，文件顶部是题目，
后面是带逐行中文注释的参考代码。

## 题量

| 目录 | 数量 | 重点 |
|---|---:|---|
| `pandas/` | 100 | CSV、索引、清洗、GroupBy、连接、重塑、时间序列和易错点 |
| `numpy/` | 100 | ndarray、广播、axis、矩阵运算、线性代数、FFT和内存 |
| `pytorch/` | 100 | Tensor、Autograd、训练、模型、CNN/RNN/Attention/Transformer |
| `python/` | 100 | 数据结构、函数、OOP、迭代器、装饰器、标准库和复杂度 |
| `ml/` | 100 | 损失、正则化、验证、指标、经典算法、漂移和可复现性 |
| `sklearn/` | 50 | Pipeline、预处理、CV、调参、模型、评估和持久化 |

“包含所有知识点”无法做数学意义上的穷尽；这里覆盖的是面试、笔试和日常 ML 开发中
最常考、最容易写错的主干知识。每个主知识点提供基础、变式、易错点和综合四种版本；
scikit-learn 每个主知识点提供基础和综合两种版本。

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

CSV 题目通过 `Path(__file__)` 定位数据，因此从仓库根目录或题库目录运行都可以。

## 自动检查

检查数量、Python 语法以及每行参考代码是否带注释：

```bash
python validate_bank.py
```

执行某一类或全部题目：

```bash
python run_all.py --category pandas
python run_all.py --category all
```

默认隐藏每题输出，只显示进度。需要观察全部结果时增加 `--show-output`。

## 重新生成

`create_bank.py` 和各个 `*_cases.py` 保存了题库定义。重新生成会删除六个题目目录并重建，
但不会删除 `data/`、README 或其他项目代码：

```bash
python create_bank.py
```
