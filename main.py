import os

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def print_title(title: str) -> None:
    # 打印分隔标题，方便你从终端里快速看出每一部分在讲什么。
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def arange_and_conversion_examples() -> None:
    # 这一部分专门演示 list、NumPy arange、PyTorch arange 的用法，以及它们之间怎么互相转换。
    print_title("list / np.arange / torch.arange / 互相转换")

    # Python list 可以直接手写元素；列表长度是 4，不是张量，没有 shape。
    python_list_manual = [0, 1, 2, 3]
    # Python list 也可以先用 range 生成，再用 list 包起来；列表长度是 5，不是张量，没有 shape。
    python_list_from_range = list(range(0, 10, 2))

    # np.arange(start, stop, step) 返回 1D NumPy 数组；shape = (5,)。
    numpy_arange_int = np.arange(0, 10, 2, dtype=np.int32)
    # np.arange 也可以生成浮点数间隔的 1D NumPy 数组；shape = (5,)。
    numpy_arange_float = np.arange(0.0, 1.0, 0.2, dtype=np.float32)

    # torch.arange(start, end, step) 返回 1D PyTorch Tensor；shape = (5,)。
    torch_arange_int = torch.arange(0, 10, 2, dtype=torch.int32)
    # torch.arange 也可以生成浮点数间隔的 1D Tensor；shape = (5,)。
    torch_arange_float = torch.arange(0.0, 1.0, 0.2, dtype=torch.float32)

    print("python_list_manual:", python_list_manual)
    print("python_list_manual length:", len(python_list_manual))
    print("python_list_from_range:", python_list_from_range)
    print("python_list_from_range length:", len(python_list_from_range))
    print("numpy_arange_int:", numpy_arange_int)
    print("numpy_arange_int.shape:", numpy_arange_int.shape)
    print("numpy_arange_float:", numpy_arange_float)
    print("numpy_arange_float.shape:", numpy_arange_float.shape)
    print("torch_arange_int:", torch_arange_int)
    print("torch_arange_int.shape:", tuple(torch_arange_int.shape))
    print("torch_arange_float:", torch_arange_float)
    print("torch_arange_float.shape:", tuple(torch_arange_float.shape))

    # list -> NumPy；转换后是 1D NumPy 数组，shape = (5,)。
    numpy_from_list = np.array(python_list_from_range, dtype=np.int32)
    # list -> PyTorch；转换后是 1D Tensor，shape = (5,)。
    torch_from_list = torch.tensor(python_list_from_range, dtype=torch.int32)

    # NumPy -> list；转换后是 Python list，长度是 5，不是张量，没有 shape。
    list_from_numpy = numpy_arange_int.tolist()
    # NumPy -> PyTorch；转换后是 1D Tensor，shape = (5,)。
    torch_from_numpy = torch.from_numpy(numpy_arange_float.copy())

    # PyTorch -> list；转换后是 Python list，长度是 5，不是张量，没有 shape。
    list_from_torch = torch_arange_int.tolist()
    # PyTorch -> NumPy；如果 Tensor 在 CPU 上，可以直接 .numpy()；转换后 shape = (5,)。
    numpy_from_torch = torch_arange_float.numpy()

    print("numpy_from_list:", numpy_from_list)
    print("numpy_from_list.shape:", numpy_from_list.shape)
    print("torch_from_list:", torch_from_list)
    print("torch_from_list.shape:", tuple(torch_from_list.shape))
    print("list_from_numpy:", list_from_numpy)
    print("list_from_numpy length:", len(list_from_numpy))
    print("torch_from_numpy:", torch_from_numpy)
    print("torch_from_numpy.shape:", tuple(torch_from_numpy.shape))
    print("list_from_torch:", list_from_torch)
    print("list_from_torch length:", len(list_from_torch))
    print("numpy_from_torch:", numpy_from_torch)
    print("numpy_from_torch.shape:", numpy_from_torch.shape)


def numpy_array_examples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # NumPy 1D 数组最像“向量”；shape = (3,)。
    numpy_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # NumPy 2D 数组最像“表格/矩阵”；shape = (2, 3)。
    numpy_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    # NumPy 3D 数组常见于“batch + sequence + feature”或“batch + H + W”；shape = (2, 3, 1)。
    numpy_3d = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=np.float32)

    print_title("NumPy 1D / 2D / 3D")
    print("numpy_1d:\n", numpy_1d)
    print("numpy_1d.shape:", numpy_1d.shape)
    print("numpy_1d.dtype:", numpy_1d.dtype)
    print("numpy_2d:\n", numpy_2d)
    print("numpy_2d.shape:", numpy_2d.shape)
    print("numpy_3d:\n", numpy_3d)
    print("numpy_3d.shape:", numpy_3d.shape)

    # zeros 常用来初始化全 0 数组；shape = (2, 3)。
    numpy_zeros = np.zeros((2, 3), dtype=np.float32)
    # ones 常用来初始化全 1 数组；shape = (2, 3)。
    numpy_ones = np.ones((2, 3), dtype=np.float32)
    # random.rand 生成 [0, 1) 均匀分布随机数；shape = (2, 3)。
    numpy_random_uniform = np.random.rand(2, 3).astype(np.float32)
    # random.randn 生成标准正态分布随机数；shape = (2, 3)。
    numpy_random_normal = np.random.randn(2, 3).astype(np.float32)
    # eye 生成单位矩阵；shape = (3, 3)。
    numpy_eye = np.eye(3, dtype=np.float32)
    # reshape 常用来改 shape 不改数据总数；这里把 (6,) 改成 (2, 3)。
    numpy_reshape_source = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)  # shape = (6,)。
    numpy_reshaped = numpy_reshape_source.reshape(2, 3)  # shape = (2, 3)。

    print("numpy_zeros.shape:", numpy_zeros.shape)
    print("numpy_ones.shape:", numpy_ones.shape)
    print("numpy_random_uniform.shape:", numpy_random_uniform.shape)
    print("numpy_random_normal.shape:", numpy_random_normal.shape)
    print("numpy_eye:\n", numpy_eye)
    print("numpy_eye.shape:", numpy_eye.shape)
    print("numpy_reshape_source.shape:", numpy_reshape_source.shape)
    print("numpy_reshaped:\n", numpy_reshaped)
    print("numpy_reshaped.shape:", numpy_reshaped.shape)
    return numpy_1d, numpy_2d, numpy_3d


def pytorch_tensor_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # PyTorch 1D Tensor；shape = (3,)。
    torch_1d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    # PyTorch 2D Tensor；shape = (2, 3)。
    torch_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    # PyTorch 3D Tensor；shape = (2, 3, 1)。
    torch_3d = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=torch.float32)

    print_title("PyTorch 1D / 2D / 3D")
    print("torch_1d:\n", torch_1d)
    print("torch_1d.shape:", tuple(torch_1d.shape))
    print("torch_1d.dtype:", torch_1d.dtype)
    print("torch_2d:\n", torch_2d)
    print("torch_2d.shape:", tuple(torch_2d.shape))
    print("torch_3d:\n", torch_3d)
    print("torch_3d.shape:", tuple(torch_3d.shape))

    # torch.zeros 生成全 0 Tensor；shape = (2, 3)。
    torch_zeros = torch.zeros((2, 3), dtype=torch.float32)
    # torch.ones 生成全 1 Tensor；shape = (2, 3)。
    torch_ones = torch.ones((2, 3), dtype=torch.float32)
    # torch.rand 生成 [0, 1) 均匀分布随机数；shape = (2, 3)。
    torch_random_uniform = torch.rand((2, 3), dtype=torch.float32)
    # torch.randn 生成标准正态分布随机数；shape = (2, 3)。
    torch_random_normal = torch.randn((2, 3), dtype=torch.float32)
    # torch.eye 生成单位矩阵；shape = (3, 3)。
    torch_eye = torch.eye(3, dtype=torch.float32)
    # reshape 常用来改张量形状；这里把 (6,) 改成 (2, 3)。
    torch_reshape_source = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)  # shape = (6,)。
    torch_reshaped = torch_reshape_source.reshape(2, 3)  # shape = (2, 3)。

    print("torch_zeros.shape:", tuple(torch_zeros.shape))
    print("torch_ones.shape:", tuple(torch_ones.shape))
    print("torch_random_uniform.shape:", tuple(torch_random_uniform.shape))
    print("torch_random_normal.shape:", tuple(torch_random_normal.shape))
    print("torch_eye:\n", torch_eye)
    print("torch_eye.shape:", tuple(torch_eye.shape))
    print("torch_reshape_source.shape:", tuple(torch_reshape_source.shape))
    print("torch_reshaped:\n", torch_reshaped)
    print("torch_reshaped.shape:", tuple(torch_reshaped.shape))
    return torch_1d, torch_2d, torch_3d


def sklearn_array_examples() -> None:
    # sklearn 通常没有自己专门的“1D/2D/3D 数组类型”，它大多数时候直接吃 NumPy 数组或 pandas DataFrame。
    dataset = load_iris()
    sklearn_data_2d = dataset.data.astype(np.float32)  # shape = (150, 4)，这是最典型的 sklearn data 形式。
    sklearn_target_1d = dataset.target.astype(np.int64)  # shape = (150,)，这是最典型的 sklearn target 形式。
    sklearn_data_3d = sklearn_data_2d.reshape(150, 4, 1)  # 这里只是演示怎么人工转成 3D；sklearn 多数模型并不直接吃 3D。

    print_title("sklearn 常见 data / target 形式")
    print("sklearn_data_2d.shape:", sklearn_data_2d.shape)
    print("sklearn_target_1d.shape:", sklearn_target_1d.shape)
    print("sklearn_data_3d.shape:", sklearn_data_3d.shape)

    # StandardScaler 一般希望输入是 2D，shape = (num_samples, num_features)。
    scaler = StandardScaler()
    scaled_data_2d = scaler.fit_transform(sklearn_data_2d)  # 输出 shape 仍然是 (150, 4)。
    print("scaled_data_2d.shape:", scaled_data_2d.shape)

    # 如果只有 1D 数据，sklearn 往往要求你先 reshape 成 2D。
    one_feature_1d = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)  # shape = (4,)。
    one_feature_2d = one_feature_1d.reshape(-1, 1)  # shape = (4, 1)。
    scaled_one_feature = scaler.fit_transform(one_feature_2d)  # 输出 shape = (4, 1)。
    print("one_feature_1d.shape:", one_feature_1d.shape)
    print("one_feature_2d.shape for sklearn:", one_feature_2d.shape)
    print("scaled_one_feature.shape:", scaled_one_feature.shape)


def conversion_examples(numpy_2d: np.ndarray, numpy_3d: np.ndarray) -> None:
    print_title("NumPy / PyTorch / pandas 互相转换")

    # NumPy -> PyTorch；shape 不变，仍然是 (2, 3)。
    torch_from_numpy = torch.from_numpy(numpy_2d)
    print("torch_from_numpy.shape:", tuple(torch_from_numpy.shape))
    print("torch_from_numpy.dtype:", torch_from_numpy.dtype)

    # PyTorch -> NumPy；如果张量在 CPU 且不带梯度，可以直接 .numpy()。
    numpy_from_torch = torch_from_numpy.numpy()
    print("numpy_from_torch.shape:", numpy_from_torch.shape)
    print("numpy_from_torch.dtype:", numpy_from_torch.dtype)

    # NumPy 3D -> pandas DataFrame 不直接自然，所以常先 reshape 成 2D。
    numpy_3d_to_2d = numpy_3d.reshape(2, 3)  # 从 (2, 3, 1) 压成 (2, 3)。
    frame_from_numpy = pd.DataFrame(numpy_3d_to_2d, columns=["feature_1", "feature_2", "feature_3"])
    print("frame_from_numpy:\n", frame_from_numpy)
    print("frame_from_numpy.shape:", frame_from_numpy.shape)

    # pandas DataFrame -> NumPy；shape = (2, 3)。
    numpy_from_frame = frame_from_numpy.to_numpy(dtype=np.float32)
    print("numpy_from_frame.shape:", numpy_from_frame.shape)

    # pandas DataFrame -> PyTorch；先转成 NumPy，再转 Tensor 更常见。
    torch_from_frame = torch.tensor(numpy_from_frame, dtype=torch.float32)
    print("torch_from_frame.shape:", tuple(torch_from_frame.shape))


def real_csv_examples(csv_path: str) -> None:
    print_title("真实 CSV 读表 -> data / target")

    # 读真实 CSV 表格；这里是 iris.csv，shape = (150, 5)。
    frame = pd.read_csv(csv_path)
    print("CSV path:", csv_path)
    print("frame.head():\n", frame.head())
    print("frame.shape:", frame.shape)
    print("frame.columns:", list(frame.columns))

    # 这里把前 4 列数值特征作为 data / X；shape = (150, 4)。
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    data = frame[feature_columns].to_numpy(dtype=np.float32)

    # 这里把最后一列 species 当作 target 原始标签；shape = (150,)。
    target_text = frame["species"].to_numpy()

    # sklearn 模型通常更喜欢数值 target，所以这里用 LabelEncoder 把字符串标签转成 0/1/2。
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target_text)  # 输出 shape = (150,)。

    print("data shape:", data.shape)
    print("target_text shape:", target_text.shape)
    print("target shape after LabelEncoder:", target.shape)
    print("label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # 做一个最常见的 train/test split。
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    print("x_train.shape:", x_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)

    # 如果后面想喂给 PyTorch，可以继续转成 Tensor。
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)  # shape = (120, 4)。
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 分类标签常用 long，shape = (120,)。
    print("x_train_tensor.shape:", tuple(x_train_tensor.shape))
    print("y_train_tensor.shape:", tuple(y_train_tensor.shape))

    # 如果后面想做 sklearn 预处理，可以直接 StandardScaler。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)  # shape = (120, 4)。
    x_test_scaled = scaler.transform(x_test)  # shape = (30, 4)。
    print("x_train_scaled.shape:", x_train_scaled.shape)
    print("x_test_scaled.shape:", x_test_scaled.shape)


def main() -> None:
    arange_and_conversion_examples()
    numpy_1d, numpy_2d, numpy_3d = numpy_array_examples()
    _ = numpy_1d  # 显式说明 1D 变量也创建过了，避免你回头看代码时以为漏了。
    _ = pytorch_tensor_examples()
    sklearn_array_examples()
    conversion_examples(numpy_2d, numpy_3d)

    csv_path = os.path.join("data", "iris.csv")
    real_csv_examples(csv_path)


if __name__ == "__main__":
    main()
