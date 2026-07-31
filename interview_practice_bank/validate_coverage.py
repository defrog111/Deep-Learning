"""检查五类面试题库是否覆盖约定的常用API与高频考点。

这里的“覆盖”表示对应术语或API必须真实存在于生成后的可学习文件中；
它和运行测试、AST重复度检查互补，避免题目可运行但课程内容被意外删减。
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parent

# 清单聚焦通用数据/ML面试，不追求枚举标准库和第三方库的每一个冷门符号。
REQUIRED: dict[str, list[str]] = {
    "python": [
        "frozenset", "defaultdict", "ChainMap", "slice(", "islice", "sorted(",
        "casefold", "*args", "**kwargs", "nonlocal", "late binding", "wraps",
        "yield from", ".send(", ".close(", "StopIteration", "contextmanager",
        "ExitStack", "finally", "自定义异常", "@dataclass", "default_factory",
        "NamedTuple", "__mro__", "super()", "ABC", "abstractmethod", "descriptor",
        "cached_property", "__hash__", "__contains__", "deepcopy", "weakref",
        "gc.collect", "re.compile", "fullmatch", "re.sub", "Path(", "csv.Dict",
        "json.loads", "pickle", "logging", "Counter", "deque", "groupby",
        "product", "lru_cache", "singledispatch", "heapq", "bisect_left",
        "Protocol", "TypedDict", "Generic", "ThreadPoolExecutor", "threading.Lock",
        "asyncio.gather", "GIL",
    ],
    "numpy": [
        "np.array", "np.asarray", "np.arange(100)", "np.arange(0, 100)",
        "np.arange(start=0, stop=100, step=1)", "np.linspace", "np.zeros",
        "np.ones", "np.full", "np.eye", "reshape", "transpose", "np.moveaxis",
        "np.newaxis", "np.take", "np.ix_", "np.broadcast_to", "np.concatenate",
        "np.stack", "np.vstack", "np.hstack", "np.block", "np.array_split",
        "keepdims", "np.add.accumulate", "np.multiply.reduce", "np.sort",
        "np.argsort", "np.partition", "np.argpartition", "np.unique", "np.bincount",
        "np.where", "np.select", "np.putmask", "np.nanmean", "np.nanmedian",
        "np.nanpercentile", "np.nan_to_num", "np.ma.masked_invalid",
        "np.random.default_rng", ".permutation(", ".shuffle(", ".choice(",
        "replace=False", ".integers(", "SeedSequence", "np.dot", "np.matmul",
        "np.tensordot", "np.linalg.solve", "np.linalg.lstsq", "np.linalg.pinv",
        "np.linalg.det", "np.linalg.slogdet", "np.linalg.cond", "np.linalg.eig",
        "np.linalg.eigh", "np.linalg.svd", "np.linalg.norm", "np.outer",
        "np.einsum", "np.kron", "np.cov", "np.corrcoef", "np.polyfit",
        "np.polynomial.Polynomial", "np.fft.rfft", "np.fft.irfft", "np.fft.fftshift",
        "np.shares_memory", "np.ascontiguousarray", "np.vectorize", "out=",
    ],
    "pytorch": [
        "torch.tensor", "torch.as_tensor", "torch.from_numpy", "torch.arange(start=",
        "dtype", "device", ".to(", ".view(", ".reshape(", ".flatten(",
        ".unflatten(", ".permute(", ".contiguous(", "torch.gather", ".scatter(",
        ".masked_fill(", "requires_grad", ".backward(", "torch.autograd.grad",
        "jacobian", "retain_grad", ".detach(", "torch.no_grad", "inference_mode",
        "nn.Parameter", "register_buffer", "nn.ModuleList", "forward_hook",
        "requires_grad_(False)", "zero_grad(set_to_none=True)", "梯度累积",
        "torch.autocast", "CrossEntropyLoss", "ignore_index", "BCEWithLogitsLoss",
        "pos_weight", "Dataset", "DataLoader", "collate_fn", "DistributedSampler",
        "TensorDataset", "torch.optim.SGD", "torch.optim.Adam", "param_groups",
        "state_dict", ".train()", ".eval()", "Dropout", "BatchNorm", "torch.save",
        "torch.load", "map_location", "xavier_uniform_", "kaiming_normal_",
        "orthogonal_", "clip_grad_norm_", "clip_grad_value_", "Embedding",
        "padding_idx", "EmbeddingBag", "pack_padded_sequence", "nn.RNN", "nn.LSTM",
        "proj_size", "nn.GRU", "GRUCell", "nn.Conv2d", "groups=", "dilation=",
        "MaxPool2d", "AdaptiveAvgPool2d", "MultiheadAttention", "key_padding_mask",
        "attn_mask", "average_attn_weights=False", "TransformerEncoder",
        "TransformerDecoder", "generate_square_subsequent_mask", "tgt_mask",
        "tgt_key_padding_mask", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR",
        ".topk(", "micro-F1",
    ],
    "ml": [
        "训练集", "验证集", "测试集", "数据泄漏", "分层切分", "Group CV",
        "nested CV", "时间序列", "标准化", "归一化", "Robust", "log1p",
        "正规方程", "Ridge", "多重共线", "残差", "梯度下降", "momentum",
        "Adam", "sigmoid", "softmax", "logits", "朴素贝叶斯", "MSE", "MAE",
        "Huber", "Pinball", "log-cosh", "L1", "L2", "Elastic Net", "Lasso",
        "偏差", "方差", "Bootstrap", "bagging", "交叉验证",
        "混淆矩阵", "precision", "recall", "specificity", "macro", "micro-F1",
        "ROC", "AUC", "PR-AUC", "类别不平衡", "过采样", "class", "阈值",
        "KNN", "维度灾难", "Gini", "entropy", "剪枝", "OOB", "AdaBoost",
        "Stacking", "SVM", "hinge", "kernel", "KMeans++", "DBSCAN", "GMM",
        "PCA", "whitening", "特征选择", "排列重要性", "校准", "Brier", "ECE",
        "Platt", "早停", "patience", "min_delta", "学习曲线", "PSI",
        "Jensen-Shannon", "漂移", "回测", "随机种子", "data_version",
    ],
    "sklearn": [
        "train_test_split", "GroupShuffleSplit", "StandardScaler", "RobustScaler",
        "SimpleImputer", "KNNImputer", "MissingIndicator", "OneHotEncoder",
        "OrdinalEncoder", "ColumnTransformer", "Pipeline", "make_pipeline",
        "FeatureUnion", "FunctionTransformer", "KFold", "StratifiedKFold",
        "GroupKFold", "TimeSeriesSplit", "cross_validate", "make_scorer",
        "GridSearchCV", "RandomizedSearchCV", "validation_curve", "learning_curve",
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "TransformedTargetRegressor",
        "LogisticRegression", "SGDClassifier", "partial_fit", "GaussianNB",
        "KNeighborsClassifier", "RadiusNeighborsClassifier", "DecisionTreeClassifier",
        "cost_complexity_pruning_path", "RandomForestClassifier", "ExtraTreesClassifier",
        "GradientBoostingClassifier", "HistGradientBoostingClassifier", "SVC",
        "LinearSVC", "CalibratedClassifierCV", "FrozenEstimator", "KMeans",
        "DBSCAN", "GaussianMixture", "IsolationForest", "PCA", "IncrementalPCA",
        "classification_report", "multilabel_confusion_matrix", "roc_auc_score",
        "precision_recall_curve", "compute_class_weight", "compute_sample_weight",
        "permutation_importance", "partial_dependence", "joblib.dump", "joblib.load",
    ],
}


def main() -> None:
    missing_by_category: dict[str, list[str]] = {}
    for category, required_tokens in REQUIRED.items():
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / category).glob("*.py"))
        )
        missing = [token for token in required_tokens if token not in source]
        if missing:
            missing_by_category[category] = missing

    if missing_by_category:
        details = "\n".join(
            f"{category}: {', '.join(tokens)}"
            for category, tokens in missing_by_category.items()
        )
        raise SystemExit(f"覆盖清单检查失败：\n{details}")

    total = sum(len(tokens) for tokens in REQUIRED.values())
    breakdown = ", ".join(
        f"{category} {len(tokens)}项" for category, tokens in REQUIRED.items()
    )
    print(f"覆盖清单检查通过：共 {total} 个必备API/考点（{breakdown}）。")


if __name__ == "__main__":
    main()
