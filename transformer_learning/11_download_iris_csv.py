"""例 11：从 UCI 官方网站下载 Iris，并整理成带表头的 CSV。

来源：https://archive.ics.uci.edu/dataset/53/iris
许可：CC BY 4.0
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
HEADER = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]


def download_iris(output_path: Path, force: bool = False) -> Path:
    if output_path.exists() and not force:
        print(f"数据已存在，跳过下载：{output_path}")
        return output_path

    print(f"正在从 UCI 下载：{DATASET_URL}")
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "transformer-learning-example/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw_lines = archive.read("iris.data").decode("utf-8").splitlines()

    rows = []
    for line in raw_lines:
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        values[-1] = values[-1].removeprefix("Iris-").lower()
        rows.append(values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"已保存 {len(rows)} 行：{output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "iris.csv",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_iris(args.output, args.force)
